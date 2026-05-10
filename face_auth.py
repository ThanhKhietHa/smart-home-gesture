"""
face_auth.py — Face Recognition + Centroid Tracker
===================================================
Changes:
  - On-screen keyboard REMOVED. Name typed via cv2 window key events.
  - handle_key() called BEFORE process_frame() every loop iteration.
  - process_frame() no longer accepts key param.
  - GRACE_FRAMES = 20 (~2 s at 10 FPS).
  - process_presence_only() for lightweight skip-frame presence check.
  - _state_recognise() draws animated scan progress bar every frame
    (smooth lerp) so HUD never looks frozen during re-scan.
  - _force_relock() resets tracker.box so status bar shows no-face immediately.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os
import time
from collections import deque

import config

GRACE_FRAMES  = 20
IOU_THRESHOLD = 0.30

_face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=getattr(config, 'FACE_DETECTION_CONFIDENCE', 0.35),
    min_face_presence_confidence=getattr(config, 'FACE_PRESENCE_CONFIDENCE', 0.35),
    output_facial_transformation_matrixes=False,
)
_landmarker = vision.FaceLandmarker.create_from_options(_face_options)

_NOSE=4; _LEFT_EYE=33; _RIGHT_EYE=263; _LEFT_MOUTH=61; _RIGHT_MOUTH=291; _CHIN=152
_STABLE=[4,6,8,9,10,33,133,159,145,263,234,454,152,13,61,291,70,300,168,197]

_ST_RECOGNISE="RECOGNISE"; _ST_NAMING="NAMING"
_ST_ENROLLING="ENROLLING"; _ST_DELETE="DELETE"


class _CentroidTracker:
    __slots__=('_box','_centroid','_id','_age')
    def __init__(self): self._box=None;self._centroid=None;self._id=0;self._age=0
    @staticmethod
    def _iou(a,b):
        xA=max(a[0],b[0]);yA=max(a[1],b[1]);xB=min(a[2],b[2]);yB=min(a[3],b[3])
        inter=max(0,xB-xA)*max(0,yB-yA)
        if inter==0: return 0.0
        return inter/float((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)
    def update(self,box):
        if box is None: self._age+=1; return self._id,False
        if self._box is None:
            self._box=box;self._centroid=((box[0]+box[2])//2,(box[1]+box[3])//2)
            self._id+=1;self._age=0;return self._id,True
        iou=self._iou(self._box,box)
        self._box=box;self._centroid=((box[0]+box[2])//2,(box[1]+box[3])//2);self._age=0
        if iou>=IOU_THRESHOLD: return self._id,False
        self._id+=1;return self._id,True
    def lost(self): self._age+=1
    def reset(self): self._box=None;self._centroid=None;self._age=0
    @property
    def age(self): return self._age
    @property
    def box(self): return self._box


def _to_np(result):
    if not result.face_landmarks: return None
    return np.array([[l.x,l.y,l.z] for l in result.face_landmarks[0]],dtype=np.float32)

def _get_box(result,shape):
    if not result.face_landmarks: return None
    H,W=shape[:2]
    xs=[l.x*W for l in result.face_landmarks[0]];ys=[l.y*H for l in result.face_landmarks[0]]
    return (int(min(xs))-10,int(min(ys))-10,int(max(xs))+10,int(max(ys))+10)

def _normalize(lm):
    nose=lm[_NOSE].copy(); eye_dist=np.linalg.norm(lm[_LEFT_EYE]-lm[_RIGHT_EYE])+1e-6
    return (lm-nose)/eye_dist

def _stable(lm_n): return lm_n[_STABLE]

def _head_pose(lm,shape):
    h,w=shape[:2]; idx=[_NOSE,_CHIN,_LEFT_EYE,_RIGHT_EYE,_LEFT_MOUTH,_RIGHT_MOUTH]
    p2d=np.array([[lm[i,0]*w,lm[i,1]*h] for i in idx],dtype=np.float32)
    cam=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float32)
    obj=np.array([[0,0,0],[0,-63.6,-12.5],[-43.3,32.7,-26],[43.3,32.7,-26],[-28.9,-28.9,-24.1],[28.9,-28.9,-24.1]],dtype=np.float32)
    _,rv,_=cv2.solvePnP(obj,p2d,cam,np.zeros((4,1),np.float32)); R,_=cv2.Rodrigues(rv)
    return float(np.degrees(np.arctan2(R[1,0],R[0,0]))),float(np.degrees(np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))))

def _shape_err(a,b): return float(np.mean(np.linalg.norm(a-b,axis=1)))
def _cosine_err(a,b):
    af,bf=a.flatten(),b.flatten()
    return float(1.0-np.dot(af,bf)/(np.linalg.norm(af)*np.linalg.norm(bf)+1e-6))

def _load_db():
    if not os.path.exists(config.ENROLLED_FILE): return {}
    try:
        with open(config.ENROLLED_FILE,'rb') as f: db=pickle.load(f)
        print(f"[FACE] Loaded {len(db)} face(s): {list(db.keys())}"); return db
    except Exception as e: print(f"[FACE] Load error: {e}"); return {}

def _save_db(db):
    with open(config.ENROLLED_FILE,'wb') as f: pickle.dump(db,f)

def _identify(lm_n,db):
    if not db: return 'Unknown',999.0,999.0,False
    st=_stable(lm_n); best_n='Unknown'; best_se=999.0; best_ie=999.0
    for name,data in db.items():
        se=_shape_err(st,data['stable']); ie=_cosine_err(st,data['stable'])
        if (0.5*se+0.5*ie)<(0.5*best_se+0.5*best_ie): best_n,best_se,best_ie=name,se,ie
    match=(best_se<config.FACE_SHAPE_THRESHOLD and best_ie<config.FACE_IDENTITY_THRESHOLD)
    return (best_n if match else 'Unknown'),best_se,best_ie,match


class FaceAuth:
    __slots__=('_db','_state','_unlocked','_unlock_name','_unlock_time',
               '_tracker','_grace_count','_tracked_id','_match_buf',
               '_last_se','_last_ie','_last_cand','_track_status',
               '_enroll_name','_enroll_col','_enroll_start','_del_names',
               '_typed','_frame_counter','_scan_prog')

    def __init__(self):
        self._db=_load_db(); self._state=_ST_RECOGNISE
        self._unlocked=False; self._unlock_name=""; self._unlock_time=0.0
        self._tracker=_CentroidTracker(); self._grace_count=0; self._tracked_id=-1
        self._match_buf=deque(maxlen=config.FACE_CONFIRM_FRAMES)
        self._last_se=999.0; self._last_ie=999.0; self._last_cand="—"
        self._track_status="SCANNING"; self._enroll_name=""
        self._enroll_col=[]; self._enroll_start=0.0; self._del_names=[]
        self._typed=""; self._frame_counter=0; self._scan_prog=0.0

    # kept so main.py setMouseCallback doesn't crash
    def mouse_callback(self,event,x,y,flags,param): pass

    def is_unlocked(self): return self._unlocked
    def unlocked_name(self): return self._unlock_name
    def enrolled_names(self): return list(self._db.keys())

    # ------------------------------------------------------------------
    # handle_key — MUST be called before process_frame each iteration
    # ------------------------------------------------------------------
    def handle_key(self, key):
        if key == -1:
            return

        if self._state == _ST_RECOGNISE:
            if   key == ord('e'): self._state=_ST_NAMING; self._typed=""
            elif key == ord('d'):
                if self._db: self._state=_ST_DELETE; self._del_names=list(self._db.keys())
            elif key == ord('r'): self._force_relock("Manual relock")

        elif self._state == _ST_NAMING:
            if   key == 27:         self._state=_ST_RECOGNISE; self._typed=""
            elif key in (13,10):
                name=self._typed.strip()
                if name: self._start_enroll(name)
                else: self._state=_ST_RECOGNISE
            elif key in (8,127):    self._typed=self._typed[:-1]
            elif 32<=key<=126:      self._typed+=chr(key).upper()

        elif self._state == _ST_DELETE:
            if   key==27:  self._state=_ST_RECOGNISE
            elif ord('1')<=key<=ord('9'):
                idx=key-ord('1')
                if idx<len(self._del_names):
                    nm=self._del_names[idx]; del self._db[nm]; _save_db(self._db)
                    print(f"[FACE] Deleted: {nm}")
                self._state=_ST_RECOGNISE; self._force_relock()

        elif self._state == _ST_ENROLLING:
            if key==27: print("[FACE] Enroll cancelled."); self._state=_ST_RECOGNISE

    # ------------------------------------------------------------------
    # process_presence_only — ~8 ms lightweight path when unlocked+skipping
    # ------------------------------------------------------------------
    def process_presence_only(self, frame):
        try:
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
            box=_get_box(result,frame.shape)
        except Exception:
            box=None
        return self._update_presence(frame,box)

    def _update_presence(self, frame, box):
        if box is not None:
            tid,_=self._tracker.update(box); self._grace_count=0
            self._track_status=f"UNLOCKED id={tid}"
            x1,y1,x2,y2=box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),3)
            cv2.putText(frame,self._unlock_name,(x1,max(y1-10,85)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        else:
            self._tracker.lost(); self._grace_count+=1
            self._track_status=f"LOST {self._grace_count}/{GRACE_FRAMES}"
            if self._unlocked and self._grace_count>=GRACE_FRAMES:
                self._force_relock("Face lost — grace expired")
        return frame

    # ------------------------------------------------------------------
    # process_frame — full path, no key param (key handled in handle_key)
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        if self._state == _ST_NAMING:
            return self._draw_naming(frame)
        if self._state == _ST_DELETE:
            return self._draw_delete(frame)

        try:
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
            lm=_to_np(result); box=_get_box(result,frame.shape)
        except Exception:
            result=None; lm=None; box=None

        if self._state == _ST_ENROLLING:
            return self._state_enrolling(frame,result,lm)
        return self._state_recognise(frame,result,lm,box)

    # ------------------------------------------------------------------
    # NAMING STATE — no keyboard widget, just overlay
    # ------------------------------------------------------------------
    def _draw_naming(self, frame):
        ov=frame.copy(); cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),(0,0,0),-1)
        cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
        cv2.putText(frame,"ENROLL NEW FACE",(20,60),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,200,255),2)
        cv2.putText(frame,"Type name, press Enter to confirm",(20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,180,180),1)
        cv2.rectangle(frame,(18,115),(frame.shape[1]-18,158),(40,40,40),-1)
        cv2.rectangle(frame,(18,115),(frame.shape[1]-18,158),(100,100,100),2)
        cv2.putText(frame,self._typed+"|",(28,148),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,200),2)
        cv2.putText(frame,"Enter = confirm     ESC = cancel",(20,188),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(120,120,120),1)
        return frame

    def _draw_delete(self, frame):
        panel=np.full_like(frame,25)
        cv2.putText(panel,"DELETE WHICH FACE?",(20,55),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,100,255),2)
        cv2.putText(panel,"Press number  |  ESC = cancel",(20,95),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(150,150,150),1)
        for i,nm in enumerate(self._del_names[:5]):
            cv2.putText(panel,f"  {i+1}.  {nm}",(20,145+i*45),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,170,0),2)
        return panel

    # ------------------------------------------------------------------
    # RECOGNISE — animated scan progress, redraws every frame
    # ------------------------------------------------------------------
    def _state_recognise(self, frame, result, lm, box):
        self._frame_counter+=1

        if self._unlocked:
            return self._update_presence(frame, box)

        # Update tracker
        if box is not None:
            track_id,_=self._tracker.update(box); self._grace_count=0
        else:
            self._tracker.lost(); self._grace_count+=1
            self._track_status="SCANNING — no face"
            track_id=-1

        # Recognition every N frames
        every_n=getattr(config,'FACE_PROCESS_EVERY_N_FRAMES_LOCKED',2)
        if self._frame_counter%every_n==0 and box is not None and lm is not None:
            H=frame.shape[0]
            face_ok=True
            if result and result.face_landmarks:
                ys=[pt.y*H for pt in result.face_landmarks[0]]
                face_ok=(max(ys)-min(ys))/H>=config.FACE_MIN_HEIGHT_FRAC
            if face_ok:
                self._run_recognition(lm,frame.shape,track_id)

        # Smooth scan progress (lerp toward real value — animates between frames)
        target=sum(self._match_buf)/max(config.FACE_CONFIRM_FRAMES,1)
        self._scan_prog+=( target-self._scan_prog)*0.3

        # Draw face box + animated progress bar under it
        if box is not None:
            x1,y1,x2,y2=box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,60,220),2)
            label=self._last_cand if self._last_cand!="—" else "Scanning..."
            cv2.putText(frame,label,(x1,max(y1-10,85)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(60,100,255),2)
            # progress bar sits just below the face box
            bw=x2-x1; by=y2+6
            cv2.rectangle(frame,(x1,by),(x2,by+6),(40,40,40),-1)
            filled=int(bw*self._scan_prog)
            if filled>0:
                cv2.rectangle(frame,(x1,by),(x1+filled,by+6),(60,120,255),-1)

        cv2.rectangle(frame,(0,0),(frame.shape[1]-1,frame.shape[0]-1),(0,0,200),4)
        return frame

    def _start_enroll(self, name):
        self._enroll_name=name; self._enroll_col=[]; self._enroll_start=time.time()
        self._state=_ST_ENROLLING; print(f"[FACE] Enrolling: {name}")

    def _state_enrolling(self, frame, result, lm):
        name=self._enroll_name; collected=self._enroll_col
        TIMEOUT=50.0; remaining=TIMEOUT-(time.time()-self._enroll_start)
        is_good=False; status="No face detected"

        if lm is not None:
            try:
                yaw,pitch=_head_pose(lm,frame.shape)
                if abs(yaw)<35 and abs(pitch)<25:
                    lm_n=_normalize(lm); collected.append(_stable(lm_n))
                    status=f"Good frame {len(collected)}/{config.FACE_ENROLL_TARGET}"; is_good=True
                    H,W=frame.shape[:2]
                    if result and result.face_landmarks:
                        for pt in result.face_landmarks[0]:
                            cv2.circle(frame,(int(pt.x*W),int(pt.y*H)),2,(0,255,0),-1)
                else:
                    status=f"Face straight — yaw={yaw:.0f} pitch={pitch:.0f}"
                    H,W=frame.shape[:2]
                    if result and result.face_landmarks:
                        for pt in result.face_landmarks[0]:
                            cv2.circle(frame,(int(pt.x*W),int(pt.y*H)),2,(0,140,255),-1)
            except Exception:
                status="Adjust position"

        cv2.rectangle(frame,(0,0),(frame.shape[1],115),(0,55,150),-1)
        cv2.putText(frame,f"ENROLLING: {name}",(20,42),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),2)
        cv2.putText(frame,status,(20,82),cv2.FONT_HERSHEY_SIMPLEX,0.75,
                    (0,255,100) if is_good else (0,150,255),2)
        prog=len(collected)/config.FACE_ENROLL_TARGET; bw=frame.shape[1]-40
        cv2.rectangle(frame,(20,90),(20+bw,108),(50,50,50),-1)
        cv2.rectangle(frame,(20,90),(20+int(bw*prog),108),(0,220,100),-1)
        cv2.putText(frame,f"{int(prog*100)}%  |  30-50 cm, face straight  |  {remaining:.0f}s left",
                    (20,125),cv2.FONT_HERSHEY_SIMPLEX,0.58,(200,200,200),1)
        cv2.putText(frame,"ESC = cancel",(20,frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,120),1)

        if len(collected)>=config.FACE_ENROLL_TARGET or remaining<=0:
            self._finish_enroll(collected,name,frame)
        return frame

    def _finish_enroll(self, collected, name, frame):
        if len(collected)>=10:
            avg=np.mean(collected,axis=0).astype(np.float32)
            self._db[name]={'stable':avg,'frames':len(collected)}; _save_db(self._db)
            cv2.imwrite(os.path.join(config.ENROLL_PHOTOS_DIR,f"{name}.jpg"),frame)
            print(f"[FACE] '{name}' enrolled ({len(collected)} frames).")
        else:
            print(f"[FACE] Enroll failed — only {len(collected)} frames.")
        self._state=_ST_RECOGNISE; self._force_relock()

    def _run_recognition(self, lm, shape, track_id):
        try: yaw,pitch=_head_pose(lm,shape)
        except Exception: return
        if abs(yaw)>55 or abs(pitch)>45: return
        lm_n=_normalize(lm); name,se,ie,match=_identify(lm_n,self._db)
        self._last_se=se; self._last_ie=ie; self._last_cand=name
        self._match_buf.append(match)
        if match and sum(self._match_buf)>=int(config.FACE_CONFIRM_FRAMES*0.7) and not self._unlocked:
            self._unlocked=True; self._unlock_name=name
            self._unlock_time=time.time(); self._tracked_id=track_id; self._scan_prog=0.0
            print(f"[FACE] UNLOCKED — {name}")

    def _force_relock(self, reason=""):
        if reason: print(f"[FACE] Relocked: {reason}")
        self._unlocked=False; self._unlock_name=""; self._tracked_id=-1
        self._grace_count=0; self._scan_prog=0.0; self._match_buf.clear()
        self._tracker.reset(); self._track_status="SCANNING"

    # ------------------------------------------------------------------
    # HUD — drawn on main thread only
    # ------------------------------------------------------------------
    def draw_status_bar(self, frame):
        cv2.rectangle(frame,(0,0),(frame.shape[1],68),
                      (0,100,0) if self._unlocked else (0,0,120),-1)
        if self._unlocked:
            rem=max(0.0,config.FACE_AUTH_TIMEOUT-(time.time()-self._unlock_time))
            cv2.putText(frame,f"UNLOCKED  [{self._unlock_name}]",(12,42),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,80),2)
            cv2.putText(frame,f"Session: {int(rem)}s",(frame.shape[1]-165,42),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(160,255,160),1)
            if self._grace_count>0:
                cv2.putText(frame,f"GRACE {self._grace_count}/{GRACE_FRAMES}",(12,62),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,255),1)
        else:
            lbl={_ST_NAMING:"NAMING — type name then Enter",
                 _ST_ENROLLING:f"ENROLLING: {self._enroll_name}",
                 _ST_DELETE:"DELETE MODE — press number"}
            if self._state in lbl:
                msg=lbl[self._state]
            elif not self._db:
                msg="NO FACES ENROLLED — press  e"
            elif self._tracker.box is None:
                msg="LOCKED — no face detected"
            else:
                msg=f"LOCKED — scanning {int(self._scan_prog*100)}%"
            cv2.putText(frame,msg,(12,42),cv2.FONT_HERSHEY_SIMPLEX,0.8,(80,140,255),2)
        return frame

    def draw_debug(self, frame):
        W=frame.shape[1]
        mc=lambda v,t:(0,220,0) if v<t else (0,80,220)
        lines=[(f"S:{self._last_se:.3f}/{config.FACE_SHAPE_THRESHOLD}",
                mc(self._last_se,config.FACE_SHAPE_THRESHOLD)),
               (f"C:{self._last_ie:.3f}/{config.FACE_IDENTITY_THRESHOLD}",
                mc(self._last_ie,config.FACE_IDENTITY_THRESHOLD)),
               (f"{self._track_status[:20]}",(180,180,180))]
        pw,ph=190,len(lines)*15+6; x0=W-pw-4; y0=72
        ov=frame.copy(); cv2.rectangle(ov,(x0,y0),(W-2,y0+ph),(15,15,15),-1)
        cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
        for i,(text,color) in enumerate(lines):
            cv2.putText(frame,text,(x0+5,y0+12+i*15),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
        return frame
