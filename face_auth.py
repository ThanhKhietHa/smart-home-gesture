"""
face_auth.py — Face Recognition + Centroid Tracker
OPTIMIZED for Jetson Orin Nano
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

# =====================================================================
# TRACKER CONSTANTS
# =====================================================================
GRACE_FRAMES    = 40
IOU_THRESHOLD   = 0.30

# =====================================================================
# MEDIAPIPE - DISABLE EXPENSIVE FEATURES
# =====================================================================
_face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
    min_face_presence_confidence=config.FACE_PRESENCE_CONFIDENCE,
    output_facial_transformation_matrixes=False,  # CRITICAL: Disable this!
)
_landmarker = vision.FaceLandmarker.create_from_options(_face_options)

# =====================================================================
# LANDMARK CONSTANTS
# =====================================================================
_NOSE        = 4
_LEFT_EYE    = 33
_RIGHT_EYE   = 263
_LEFT_MOUTH  = 61
_RIGHT_MOUTH = 291
_CHIN        = 152

_STABLE = [4,6,8,9,10, 33,133,159,145,263, 234,454,
           152,13, 61,291, 70,300, 168,197]

# =====================================================================
# APP STATES
# =====================================================================
_ST_RECOGNISE = "RECOGNISE"
_ST_TYPING    = "TYPING"
_ST_ENROLLING = "ENROLLING"
_ST_DELETE    = "DELETE"

# =====================================================================
# CENTROID TRACKER
# =====================================================================
class _CentroidTracker:
    __slots__ = ('_box', '_centroid', '_id', '_age')
    
    def __init__(self):
        self._box      = None
        self._centroid = None
        self._id       = 0
        self._age      = 0

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        if inter == 0:
            return 0.0
        aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / float(aA + aB - inter + 1e-6)

    def update(self, box):
        if box is None:
            self._age += 1
            return self._id, False

        if self._box is None:
            self._box      = box
            self._centroid = ((box[0]+box[2])//2, (box[1]+box[3])//2)
            self._id      += 1
            self._age      = 0
            return self._id, True

        iou = self._iou(self._box, box)

        if iou >= IOU_THRESHOLD:
            self._box      = box
            self._centroid = ((box[0]+box[2])//2, (box[1]+box[3])//2)
            self._age      = 0
            return self._id, False
        else:
            self._box      = box
            self._centroid = ((box[0]+box[2])//2, (box[1]+box[3])//2)
            self._id      += 1
            self._age      = 0
            return self._id, True

    def lost(self):
        self._age += 1

    def reset(self):
        self._box      = None
        self._centroid = None
        self._age      = 0

    @property
    def age(self):
        return self._age

    @property
    def box(self):
        return self._box

# =====================================================================
# LANDMARK HELPERS
# =====================================================================
def _to_np(result):
    if not result.face_landmarks:
        return None
    return np.array([[l.x,l.y,l.z] for l in result.face_landmarks[0]],
                    dtype=np.float32)

def _get_box(result, shape):
    if not result.face_landmarks:
        return None
    H, W = shape[:2]
    xs = [l.x*W for l in result.face_landmarks[0]]
    ys = [l.y*H for l in result.face_landmarks[0]]
    return (int(min(xs))-10, int(min(ys))-10,
            int(max(xs))+10, int(max(ys))+10)

def _normalize(lm):
    nose     = lm[_NOSE].copy()
    eye_dist = np.linalg.norm(lm[_LEFT_EYE]-lm[_RIGHT_EYE]) + 1e-6
    return (lm - nose) / eye_dist

def _stable(lm_n):
    return lm_n[_STABLE]

def _head_pose(lm, shape):
    h, w  = shape[:2]
    idx   = [_NOSE,_CHIN,_LEFT_EYE,_RIGHT_EYE,_LEFT_MOUTH,_RIGHT_MOUTH]
    p2d   = np.array([[lm[i,0]*w, lm[i,1]*h] for i in idx], dtype=np.float32)
    cam   = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float32)
    obj   = np.array([[0,0,0],[0,-63.6,-12.5],[-43.3,32.7,-26],
                      [43.3,32.7,-26],[-28.9,-28.9,-24.1],[28.9,-28.9,-24.1]],
                     dtype=np.float32)
    _, rv, _ = cv2.solvePnP(obj, p2d, cam, np.zeros((4,1),np.float32))
    R, _     = cv2.Rodrigues(rv)
    yaw   = float(np.degrees(np.arctan2(R[1,0],R[0,0])))
    pitch = float(np.degrees(np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))))
    return yaw, pitch

def _shape_err(a, b):
    return float(np.mean(np.linalg.norm(a-b, axis=1)))

def _cosine_err(a, b):
    af, bf = a.flatten(), b.flatten()
    return float(1.0 - np.dot(af,bf)/(np.linalg.norm(af)*np.linalg.norm(bf)+1e-6))

# =====================================================================
# DATABASE
# =====================================================================
def _load_db():
    if not os.path.exists(config.ENROLLED_FILE):
        return {}
    try:
        with open(config.ENROLLED_FILE,'rb') as f:
            db = pickle.load(f)
        print(f"[FACE] Loaded {len(db)} enrolled face(s)")
        return db
    except Exception as e:
        print(f"[FACE] Could not load enrolled faces: {e}")
        return {}

def _save_db(db):
    with open(config.ENROLLED_FILE,'wb') as f:
        pickle.dump(db, f)

def _identify(lm_n, db):
    if not db:
        return 'Unknown', 999.0, 999.0, False
    st      = _stable(lm_n)
    best_n  = 'Unknown'
    best_se = 999.0
    best_ie = 999.0
    for name, data in db.items():
        se = _shape_err(st, data['stable'])
        ie = _cosine_err(st, data['stable'])
        if (0.5*se+0.5*ie) < (0.5*best_se+0.5*best_ie):
            best_n, best_se, best_ie = name, se, ie
    match = (best_se < config.FACE_SHAPE_THRESHOLD and
             best_ie < config.FACE_IDENTITY_THRESHOLD)
    if not match:
        best_n = 'Unknown'
    return best_n, best_se, best_ie, match

# =====================================================================
# KEYBOARD UI (MINIMAL)
# =====================================================================
_KB_ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM") + ["DEL", "OK"],
]
_KB_H = 165

def _draw_keyboard(frame, typed):
    h, w  = frame.shape[:2]
    panel = np.full((_KB_H, w, 3), 30, dtype=np.uint8)
    cv2.rectangle(panel, (10,5), (w-10,38), (50,50,50), -1)
    cv2.putText(panel, typed+"|", (16,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,200), 2)
    KH = 34
    for ri, row in enumerate(_KB_ROWS):
        n    = len(row)
        kw   = (w-20)//10
        yt   = 44 + ri*(KH+5)
        xoff = (w - n*(kw+4))//2
        for ci, ch in enumerate(row):
            kx   = xoff + ci*(kw+4)
            bw   = kw*2 if ch in ("DEL","OK") else kw
            bg   = (0,110,0) if ch=="OK" else (110,0,0) if ch=="DEL" else (65,65,88)
            cv2.rectangle(panel,(kx,yt),(kx+bw,yt+KH),bg,-1)
            cv2.rectangle(panel,(kx,yt),(kx+bw,yt+KH),(130,130,130),1)
            tx = kx + max((bw-len(ch)*8)//2, 2)
            cv2.putText(panel, ch,(tx,yt+23),
                        cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
    return np.vstack([frame, panel])

def _kb_hittest(fh, fw, mx, my, typed):
    if my < fh:
        return typed, False, False
    ry = my - fh
    KH = 34
    for ri, row in enumerate(_KB_ROWS):
        n    = len(row)
        kw   = (fw-20)//10
        yt   = 44 + ri*(KH+5)
        if not (yt <= ry <= yt+KH):
            continue
        xoff = (fw - n*(kw+4))//2
        for ci, ch in enumerate(row):
            kx = xoff + ci*(kw+4)
            bw = kw*2 if ch in ("DEL","OK") else kw
            if kx <= mx <= kx+bw:
                if ch == "DEL":  return typed[:-1], False, False
                elif ch == "OK": return typed, bool(typed.strip()), False
                else:            return typed+ch,   False,          False
    return typed, False, False

# =====================================================================
# FACEAUTH CLASS
# =====================================================================
class FaceAuth:
    __slots__ = ('_db', '_state', '_unlocked', '_unlock_name', '_unlock_time',
                 '_tracker', '_grace_count', '_tracked_id', '_match_buf',
                 '_last_se', '_last_ie', '_last_cand', '_track_status',
                 '_enroll_name', '_enroll_col', '_enroll_start', '_del_names',
                 '_typed', '_mx', '_my', '_clicked', '_frame_counter')
    
    def __init__(self):
        self._db           = _load_db()
        self._state        = _ST_RECOGNISE
        self._unlocked     = False
        self._unlock_name  = ""
        self._unlock_time  = 0.0
        self._tracker      = _CentroidTracker()
        self._grace_count  = 0
        self._tracked_id   = -1
        self._match_buf    = deque(maxlen=config.FACE_CONFIRM_FRAMES)
        self._last_se      = 999.0
        self._last_ie      = 999.0
        self._last_cand    = "—"
        self._track_status = "SCANNING"
        self._enroll_name  = ""
        self._enroll_col   = []
        self._enroll_start = 0.0
        self._del_names    = []
        self._typed        = ""
        self._mx = 0
        self._my = 0
        self._clicked = False
        self._frame_counter = 0

    def mouse_callback(self, event, x, y, flags, param):
        self._mx, self._my = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked = True

    def is_unlocked(self) -> bool:
        return self._unlocked
    
    def unlocked_name(self) -> str:
        return self._unlock_name
    
    def enrolled_names(self) -> list:
        return list(self._db.keys())

    def handle_key(self, key):
        if self._state != _ST_RECOGNISE:
            return
        if key == ord('e'):
            self._state = _ST_TYPING
            self._typed = ""
        elif key == ord('d'):
            if self._db:
                self._state = _ST_DELETE
                self._del_names = list(self._db.keys())
        elif key == ord('r'):
            self._force_relock("Manual relock")

    def process_frame(self, frame, key):
        if self._state == _ST_TYPING:
            return self._state_typing(frame, key)
        if self._state == _ST_DELETE:
            return self._state_delete(frame, key)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        lm = _to_np(result)
        box = _get_box(result, frame.shape)

        if self._state == _ST_ENROLLING:
            return self._state_enrolling(frame, result, lm, key)

        return self._state_recognise(frame, result, lm, box, key)

    def _state_recognise(self, frame, result, lm, box, key):
        self._frame_counter += 1
        should_detect = False
        
        if not self._unlocked:
            should_detect = (self._frame_counter % config.FACE_DETECT_EVERY_N_FRAMES == 0)
        else:
            should_detect = (self._frame_counter % config.FACE_DETECT_EVERY_N_FRAMES_UNLOCKED == 0)
        
        # Use cached box for skipped frames
        if not should_detect and self._tracker.box is not None:
            box = self._tracker.box

        if box is not None:
            track_id, is_new = self._tracker.update(box)
            self._grace_count = 0

            H = frame.shape[0]
            face_ok = True
            if result and result.face_landmarks and should_detect:
                ys = [pt.y*H for pt in result.face_landmarks[0]]
                face_ok = (max(ys)-min(ys))/H >= config.FACE_MIN_HEIGHT_FRAC

            run_recognition = (not self._unlocked) and face_ok and (lm is not None) and should_detect

            if run_recognition:
                self._run_recognition(lm, frame.shape, track_id)

            x1, y1, x2, y2 = box
            if self._unlocked:
                box_color = (0, 220, 0)
                thickness = 3
                name_label = self._unlock_name
                label_col = (0, 255, 0)
            else:
                box_color = (0, 60, 220)
                thickness = 2
                name_label = self._last_cand if self._last_cand != "—" else "Scan..."
                label_col = (60, 100, 255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, thickness)
            cv2.putText(frame, name_label, (x1, max(y1-10, 85)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_col, 2)

            self._track_status = (
                f"UNLOCKED" if self._unlocked
                else f"SCAN"
            )

        else:
            self._tracker.lost()
            self._grace_count += 1

            if self._unlocked and self._grace_count >= GRACE_FRAMES:
                self._force_relock("Face lost")

        if (self._unlocked and 
            (time.time()-self._unlock_time) > config.FACE_AUTH_TIMEOUT):
            self._force_relock("Timeout")

        if not self._unlocked:
            cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1),
                         (0,0,200), 3)

        return frame

    def _run_recognition(self, lm, shape, track_id):
        try:
            yaw, pitch = _head_pose(lm, shape)
        except:
            return

        if abs(yaw) > 55 or abs(pitch) > 45:
            return

        lm_n = _normalize(lm)
        name, se, ie, match = _identify(lm_n, self._db)

        self._last_se = se
        self._last_ie = ie
        self._last_cand = name
        self._match_buf.append(match)

        if (match and sum(self._match_buf) >= int(config.FACE_CONFIRM_FRAMES * 0.7)
                and not self._unlocked):
            self._unlocked = True
            self._unlock_name = name
            self._unlock_time = time.time()
            self._tracked_id = track_id
            print(f"[FACE] UNLOCKED — {name}")

    def _force_relock(self, reason=""):
        if reason:
            print(f"[FACE] Re-locked: {reason}")
        self._unlocked = False
        self._unlock_name = ""
        self._tracked_id = -1
        self._grace_count = 0
        self._match_buf.clear()
        self._tracker.reset()

    def _state_typing(self, frame, key):
        combined = _draw_keyboard(frame, self._typed)
        fh, fw = frame.shape[:2]
        if self._clicked:
            self._typed, done, _ = _kb_hittest(fh, fw, self._mx, self._my, self._typed)
            self._clicked = False
            if done and self._typed.strip():
                self._start_enroll(self._typed.strip())
        if key == 27:
            self._state = _ST_RECOGNISE
        elif key in (13,10):
            if self._typed.strip():
                self._start_enroll(self._typed.strip())
        elif key == 8:
            self._typed = self._typed[:-1]
        elif 32 <= key <= 122:
            self._typed += chr(key).upper()
        return combined

    def _start_enroll(self, name):
        self._enroll_name = name
        self._enroll_col = []
        self._enroll_start = time.time()
        self._state = _ST_ENROLLING
        print(f"[FACE] Enrolling: {name}")

    def _state_enrolling(self, frame, result, lm, key):
        name = self._enroll_name
        collected = self._enroll_col
        remaining = 50.0 - (time.time() - self._enroll_start)
        is_good = False
        status = "No face"

        if lm is not None:
            try:
                yaw, pitch = _head_pose(lm, frame.shape)
                if abs(yaw) < 35 and abs(pitch) < 25:
                    lm_n = _normalize(lm)
                    collected.append(_stable(lm_n))
                    status = f"Good {len(collected)}/{config.FACE_ENROLL_TARGET}"
                    is_good = True
            except:
                status = "Adjust"

        cv2.rectangle(frame, (0,0), (frame.shape[1],100), (0,55,150), -1)
        cv2.putText(frame, f"ENROLL: {name}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, status, (20,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,100) if is_good else (0,150,255), 2)
        
        prog = len(collected)/config.FACE_ENROLL_TARGET
        bw = frame.shape[1]-40
        cv2.rectangle(frame, (20,90), (20+bw,110), (50,50,50), -1)
        cv2.rectangle(frame, (20,90), (20+int(bw*prog),110), (0,220,100), -1)

        if key == 27:
            self._state = _ST_RECOGNISE
        elif len(collected) >= config.FACE_ENROLL_TARGET or remaining <= 0:
            self._finish_enroll(collected, name, frame)

        return frame

    def _finish_enroll(self, collected, name, frame):
        if len(collected) >= 10:
            avg = np.mean(collected, axis=0).astype(np.float32)
            self._db[name] = {'stable': avg, 'frames': len(collected)}
            _save_db(self._db)
            photo = os.path.join(config.ENROLL_PHOTOS_DIR, f"{name}.jpg")
            cv2.imwrite(photo, frame)
            print(f"[FACE] '{name}' enrolled")
        else:
            print(f"[FACE] Enroll failed - only {len(collected)} frames")
        self._state = _ST_RECOGNISE
        self._force_relock()

    def _state_delete(self, frame, key):
        panel = np.full_like(frame, 25)
        cv2.putText(panel, "DELETE WHICH FACE?", (20,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,100,255), 2)
        for i, nm in enumerate(self._del_names[:5]):
            cv2.putText(panel, f"  {i+1}. {nm}", (20,100+i*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,170,0), 2)
        if key == 27:
            self._state = _ST_RECOGNISE
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(self._del_names):
                nm = self._del_names[idx]
                del self._db[nm]
                _save_db(self._db)
                print(f"[FACE] Deleted: {nm}")
            self._state = _ST_RECOGNISE
            self._force_relock()
        return panel

    def draw_status_bar(self, frame):
        cv2.rectangle(frame, (0,0), (frame.shape[1],55),
                      (0,100,0) if self._unlocked else (0,0,120), -1)
        if self._unlocked:
            rem = max(0.0, config.FACE_AUTH_TIMEOUT - (time.time() - self._unlock_time))
            cv2.putText(frame, f"UNLOCKED [{self._unlock_name}]", (10,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,80), 2)
        else:
            if not self._db:
                msg = "NO FACES - press 'e'"
            else:
                prog = sum(self._match_buf)/max(config.FACE_CONFIRM_FRAMES,1)
                msg = f"LOCKED [{int(prog*100)}%]"
            cv2.putText(frame, msg, (10,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,140,255), 2)
        return frame

    def draw_debug(self, frame):
        H, W = frame.shape[:2]
        lines = [
            (f"Shape:{self._last_se:.3f}", (0,220,0) if self._last_se < config.FACE_SHAPE_THRESHOLD else (0,80,220)),
            (f"Cos:{self._last_ie:.4f}", (0,220,0) if self._last_ie < config.FACE_IDENTITY_THRESHOLD else (0,80,220)),
        ]
        x0 = W - 130
        y0 = 60
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (x0, y0 + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return frame
