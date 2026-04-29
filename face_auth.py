"""
face_auth.py — Face Recognition & Enrollment Module
=====================================================
Responsibilities:
  - Enroll faces (on-screen keyboard, no input(), no camera hijack)
  - Recognize faces frame-by-frame
  - Expose is_unlocked() / unlocked_name() to main.py
  - Manage auth state: unlock, re-lock, timeout

State machine:
  RECOGNISE → TYPING → ENROLLING → RECOGNISE
              (on-screen keyboard)
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
# MEDIAPIPE
# =====================================================================
_face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.4,
    min_face_presence_confidence=0.4,
    output_facial_transformation_matrixes=True,
)
_landmarker = vision.FaceLandmarker.create_from_options(_face_options)

# =====================================================================
# LANDMARK CONSTANTS
# =====================================================================
_NOSE       = 4
_LEFT_EYE   = 33
_RIGHT_EYE  = 263
_LEFT_MOUTH = 61
_RIGHT_MOUTH= 291
_CHIN       = 152

_STABLE = [4,6,8,9,10, 33,133,159,145,263, 234,454,
           152,10, 61,291, 70,300, 168,197]

# =====================================================================
# APP STATES
# =====================================================================
_ST_RECOGNISE  = "RECOGNISE"
_ST_TYPING     = "TYPING"
_ST_ENROLLING  = "ENROLLING"
_ST_DELETE     = "DELETE"

# =====================================================================
# ON-SCREEN KEYBOARD
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
    cv2.putText(panel, "Type name then OK  |  ESC=cancel",
                (w-340,26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150,150,150), 1)
    KH = 34
    for ri, row in enumerate(_KB_ROWS):
        n    = len(row)
        kw   = (w-20)//10
        yt   = 44 + ri*(KH+5)
        xoff = (w - n*(kw+4))//2
        for ci, ch in enumerate(row):
            kx = xoff + ci*(kw+4)
            bw = kw*2 if ch in ("DEL","OK") else kw
            bg = (0,110,0) if ch=="OK" else (110,0,0) if ch=="DEL" else (65,65,88)
            cv2.rectangle(panel,(kx,yt),(kx+bw,yt+KH),bg,-1)
            cv2.rectangle(panel,(kx,yt),(kx+bw,yt+KH),(130,130,130),1)
            tx = kx + max((bw-len(ch)*8)//2, 2)
            cv2.putText(panel, ch,(tx,yt+23),
                        cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1)
    cv2.putText(panel,"Click keys  |  Enter=confirm  |  ESC=cancel",
                (10,_KB_H-6), cv2.FONT_HERSHEY_SIMPLEX,0.45,(130,130,130),1)
    return np.vstack([frame, panel])

def _kb_hittest(fh, fw, mx, my, typed):
    """Returns (typed, done, cancelled)."""
    if my < fh:
        return typed, False, False
    ry = my - fh
    KH = 34
    for ri, row in enumerate(_KB_ROWS):
        n    = len(row)
        kw   = (fw-20)//10
        yt   = 44 + ri*(KH+5)
        yb   = yt+KH
        if not (yt <= ry <= yb):
            continue
        xoff = (fw - n*(kw+4))//2
        for ci, ch in enumerate(row):
            kx = xoff + ci*(kw+4)
            bw = kw*2 if ch in ("DEL","OK") else kw
            if kx <= mx <= kx+bw:
                if ch == "DEL":   return typed[:-1], False, False
                elif ch == "OK":  return typed, bool(typed.strip()), False
                else:             return typed+ch,   False,           False
    return typed, False, False

# =====================================================================
# LANDMARK HELPERS
# =====================================================================
def _to_np(result):
    if not result.face_landmarks:
        return None
    return np.array([[l.x,l.y,l.z] for l in result.face_landmarks[0]],
                    dtype=np.float32)

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
        print(f"[FACE] Loaded {len(db)} enrolled face(s): {list(db.keys())}")
        return db
    except Exception as e:
        print(f"[FACE] Could not load enrolled faces: {e}")
        return {}

def _save_db(db):
    with open(config.ENROLLED_FILE,'wb') as f:
        pickle.dump(db, f)

def _identify(lm_n, db):
    """Returns (name, shape_err, cosine_err, is_match)."""
    if not db:
        return 'Unknown', 999.0, 999.0, False
    st       = _stable(lm_n)
    best_n   = 'Unknown'
    best_se  = 999.0
    best_ie  = 999.0
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
# FACEAUTH CLASS — used by main.py
# =====================================================================
class FaceAuth:
    """
    Call process_frame(frame) every loop iteration.
    Read is_unlocked() and unlocked_name() to gate gestures.
    Press 'e' / 'd' / 'r' keys via handle_key(key).
    """

    def __init__(self):
        self._db           = _load_db()
        self._state        = _ST_RECOGNISE

        # Auth state
        self._unlocked     = False
        self._unlock_name  = ""
        self._unlock_time  = 0.0
        self._match_buf    = deque(maxlen=config.FACE_CONFIRM_FRAMES)
        self._relock_buf   = deque(maxlen=config.FACE_RELOCK_FRAMES)

        # Metrics for debug panel
        self.last_se       = 999.0
        self.last_ie       = 999.0
        self.last_cand     = "—"

        # Enroll state
        self._enroll_name  = ""
        self._enroll_col   = []
        self._enroll_start = 0.0

        # Delete state
        self._del_names    = []

        # Typing state
        self._typed        = ""

        # Mouse
        self._mx = 0
        self._my = 0
        self._clicked = False

    # ── Mouse callback (register with cv2.setMouseCallback) ───────────
    def mouse_callback(self, event, x, y, flags, param):
        self._mx, self._my = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked = True

    # ── Public queries ─────────────────────────────────────────────────
    def is_unlocked(self) -> bool:
        return self._unlocked

    def unlocked_name(self) -> str:
        return self._unlock_name

    def enrolled_names(self) -> list:
        return list(self._db.keys())

    # ── Key handler (called from main loop) ───────────────────────────
    def handle_key(self, key: int):
        if self._state != _ST_RECOGNISE:
            return   # keys during enroll/type handled inside process_frame
        if key == ord('e'):
            self._state  = _ST_TYPING
            self._typed  = ""
            self._clicked = False
        elif key == ord('d'):
            if self._db:
                self._state     = _ST_DELETE
                self._del_names = list(self._db.keys())
        elif key == ord('r'):
            self._unlocked = False
            self._match_buf.clear()
            self._relock_buf.clear()
            print("[FACE] Manually re-locked.")

    # ── Main per-frame call ────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray, key: int) -> np.ndarray:
        """
        Runs face detection + auth logic on `frame`.
        Draws all face UI onto frame.
        Returns (possibly combined) frame to display.
        """
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        lm     = _to_np(result)

        # Face size check
        face_ok = False
        face_too_small = False
        if lm is not None:
            H = frame.shape[0]
            ys = [pt.y*H for pt in result.face_landmarks[0]]
            if (max(ys)-min(ys))/H >= config.FACE_MIN_HEIGHT_FRAC:
                face_ok = True
            else:
                face_too_small = True

        # ── Route to correct state ─────────────────────────────────────
        if self._state == _ST_RECOGNISE:
            return self._state_recognise(frame, result, lm, face_ok,
                                          face_too_small, key)
        elif self._state == _ST_TYPING:
            return self._state_typing(frame, key)
        elif self._state == _ST_ENROLLING:
            return self._state_enrolling(frame, result, lm, key)
        elif self._state == _ST_DELETE:
            return self._state_delete(frame, key)

        return frame

    # ── STATE: RECOGNISE ──────────────────────────────────────────────
    def _state_recognise(self, frame, result, lm, face_ok,
                          face_too_small, key):
        frame_match  = False
        matched_name = "Unknown"
        reject_msg   = ""

        if face_ok and self._db:
            try:
                yaw, pitch = _head_pose(lm, frame.shape)
                if abs(yaw) < 55 and abs(pitch) < 45:
                    lm_n = _normalize(lm)
                    matched_name, self.last_se, self.last_ie, frame_match = \
                        _identify(lm_n, self._db)
                    self.last_cand = matched_name
                else:
                    reject_msg = "Turn head straight"
            except Exception:
                pass
        elif face_too_small:
            reject_msg = "Move closer (30-50cm)"

        self._match_buf.append(frame_match)
        self._relock_buf.append(frame_match)

        # Unlock
        if (sum(self._match_buf) >= int(config.FACE_CONFIRM_FRAMES*0.7)
                and not self._unlocked):
            self._unlocked    = True
            self._unlock_name = matched_name
            self._unlock_time = time.time()
            print(f"[FACE] UNLOCKED — {self._unlock_name}")

        # Re-lock: no face
        if (self._unlocked
                and len(self._relock_buf) == config.FACE_RELOCK_FRAMES
                and sum(self._relock_buf) == 0):
            self._unlocked = False
            print("[FACE] Re-locked (face gone)")

        # Re-lock: timeout
        if (self._unlocked
                and (time.time()-self._unlock_time) > config.FACE_AUTH_TIMEOUT):
            self._unlocked = False
            print("[FACE] Re-locked (timeout)")

        # Draw landmarks + box
        if lm is not None:
            H, W = frame.shape[:2]
            col  = (0,255,0) if frame_match else (0,70,220)
            for pt in result.face_landmarks[0]:
                cv2.circle(frame,(int(pt.x*W),int(pt.y*H)),1,col,-1)
            xs  = [pt.x*W for pt in result.face_landmarks[0]]
            ys2 = [pt.y*H for pt in result.face_landmarks[0]]
            x1,y1 = int(min(xs))-10, int(min(ys2))-10
            x2,y2 = int(max(xs))+10, int(max(ys2))+10
            cv2.rectangle(frame,(x1,y1),(x2,y2),
                          (0,220,0) if frame_match else (0,60,220),
                          3 if frame_match else 2)
            lbl = (reject_msg if reject_msg
                   else matched_name if self._db
                   else "No faces enrolled")
            lc  = ((0,150,255) if reject_msg
                   else (0,255,0) if frame_match
                   else (60,100,255))
            cv2.putText(frame, lbl, (x1, max(y1-10,85)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, lc, 2)

        # Locked red overlay
        if not self._unlocked:
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),(0,0,110),-1)
            cv2.addWeighted(ov,0.18,frame,0.82,0,frame)
            cv2.rectangle(frame,(0,0),(frame.shape[1]-1,frame.shape[0]-1),(0,0,190),4)

        return frame

    # ── STATE: TYPING ─────────────────────────────────────────────────
    def _state_typing(self, frame, key):
        combined = _draw_keyboard(frame, self._typed)
        fh, fw   = frame.shape[:2]

        if self._clicked:
            self._typed, done, _ = _kb_hittest(
                fh, fw, self._mx, self._my, self._typed)
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
        self._enroll_name  = name
        self._enroll_col   = []
        self._enroll_start = time.time()
        self._state        = _ST_ENROLLING
        print(f"[FACE] Enrolling: {name}")

    # ── STATE: ENROLLING ──────────────────────────────────────────────
    def _state_enrolling(self, frame, result, lm, key):
        name      = self._enroll_name
        collected = self._enroll_col
        TIMEOUT   = 50.0
        remaining = TIMEOUT - (time.time()-self._enroll_start)
        is_good   = False
        status    = "No face detected"

        if lm is not None:
            try:
                yaw, pitch = _head_pose(lm, frame.shape)
                if abs(yaw) < 35 and abs(pitch) < 25:
                    lm_n = _normalize(lm)
                    collected.append(_stable(lm_n))
                    status  = f"Good frame {len(collected)}/{config.FACE_ENROLL_TARGET}"
                    is_good = True
                    H, W = frame.shape[:2]
                    for pt in result.face_landmarks[0]:
                        cv2.circle(frame,(int(pt.x*W),int(pt.y*H)),2,(0,255,0),-1)
                else:
                    status = f"Turn straight  yaw={yaw:.0f}  pitch={pitch:.0f}"
                    H, W = frame.shape[:2]
                    for pt in result.face_landmarks[0]:
                        cv2.circle(frame,(int(pt.x*W),int(pt.y*H)),2,(0,140,255),-1)
            except Exception:
                status = "Adjust position"

        cv2.rectangle(frame,(0,0),(frame.shape[1],115),(0,55,150),-1)
        cv2.putText(frame,f"ENROLLING: {name}",
                    (20,42),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
        cv2.putText(frame,status,(20,82),cv2.FONT_HERSHEY_SIMPLEX,0.85,
                    (0,255,100) if is_good else (0,150,255),2)
        prog = len(collected)/config.FACE_ENROLL_TARGET
        bw   = frame.shape[1]-40
        cv2.rectangle(frame,(20,118),(20+bw,140),(50,50,50),-1)
        cv2.rectangle(frame,(20,118),(20+int(bw*prog),140),(0,220,100),-1)
        cv2.putText(frame,
                    f"{int(prog*100)}%  —  30-50cm, look straight  |  {remaining:.0f}s left",
                    (20,162),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1)
        cv2.putText(frame,"ESC = cancel",
                    (20,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(150,150,150),1)

        if key == 27:
            print("[FACE] Enrollment cancelled.")
            self._state = _ST_RECOGNISE
            self._reset_buffers()
        elif (len(collected) >= config.FACE_ENROLL_TARGET or remaining <= 0):
            self._finish_enroll(collected, name, frame)

        return frame

    def _finish_enroll(self, collected, name, frame):
        if len(collected) >= 10:
            avg = np.mean(collected, axis=0).astype(np.float32)
            self._db[name] = {'stable': avg, 'frames': len(collected)}
            _save_db(self._db)
            photo = os.path.join(config.ENROLL_PHOTOS_DIR, f"{name}.jpg")
            cv2.imwrite(photo, frame)
            print(f"[FACE] '{name}' enrolled ({len(collected)} frames). Photo: {photo}")
        else:
            print(f"[FACE] Enroll failed — only {len(collected)} frames.")
            print("  Tips: 30-50cm, good lighting, face straight.")
        self._state = _ST_RECOGNISE
        self._reset_buffers()

    # ── STATE: DELETE ─────────────────────────────────────────────────
    def _state_delete(self, frame, key):
        panel = np.full_like(frame, 25)
        cv2.putText(panel,"DELETE WHICH FACE?",(20,55),
                    cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,100,255),2)
        cv2.putText(panel,"Press number key  |  ESC = cancel",
                    (20,95),cv2.FONT_HERSHEY_SIMPLEX,0.7,(150,150,150),1)
        for i, nm in enumerate(self._del_names):
            cv2.putText(panel,f"  {i+1}.  {nm}",(20,145+i*45),
                        cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,170,0),2)

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
            self._reset_buffers()

        return panel

    def _reset_buffers(self):
        self._match_buf.clear()
        self._relock_buf.clear()

    # ── Draw auth status bar (called by main.py on top of combined frame)
    def draw_status_bar(self, frame: np.ndarray) -> np.ndarray:
        """Draws the LOCKED/UNLOCKED bar at top. Call after process_frame."""
        cv2.rectangle(frame,(0,0),(frame.shape[1],72),
                      (0,100,0) if self._unlocked else (0,0,120),-1)
        if self._unlocked:
            rem = max(0.0, config.FACE_AUTH_TIMEOUT-(time.time()-self._unlock_time))
            cv2.putText(frame,f"UNLOCKED  [ {self._unlock_name} ]",
                        (15,48),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,80),2)
            cv2.putText(frame,f"Session: {int(rem)}s",
                        (frame.shape[1]-170,48),
                        cv2.FONT_HERSHEY_SIMPLEX,0.75,(160,255,160),1)
        else:
            if not self._db:
                msg = "NO FACES ENROLLED — press 'e'"
            elif not self._match_buf:
                msg = "LOCKED — No face in view"
            else:
                prog = sum(self._match_buf)/config.FACE_CONFIRM_FRAMES
                msg  = f"LOCKED — Scanning... {int(prog*100)}%"
            cv2.putText(frame, msg,(15,48),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(80,140,255),2)

        # Scan progress bar
        if not self._unlocked and self._match_buf:
            prog = sum(self._match_buf)/config.FACE_CONFIRM_FRAMES
            bw   = frame.shape[1]-40
            cv2.rectangle(frame,(20,76),(20+bw,92),(50,50,50),-1)
            cv2.rectangle(frame,(20,76),
                          (20+int(bw*min(prog,1.0)),92),(0,180,255),-1)
        return frame

    # ── Draw debug panel (bottom of frame) ────────────────────────────
    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        my = frame.shape[0]-105
        cv2.rectangle(frame,(0,my-8),(360,frame.shape[0]),(20,20,20),-1)
        mc = lambda v,t: (0,200,0) if v<t else (0,50,220)
        cv2.putText(frame,
            f"Shape : {self.last_se:.3f}  thr={config.FACE_SHAPE_THRESHOLD}",
            (8,my+16),cv2.FONT_HERSHEY_SIMPLEX,0.56,
            mc(self.last_se,config.FACE_SHAPE_THRESHOLD),1)
        cv2.putText(frame,
            f"Cosine: {self.last_ie:.3f}  thr={config.FACE_IDENTITY_THRESHOLD}",
            (8,my+36),cv2.FONT_HERSHEY_SIMPLEX,0.56,
            mc(self.last_ie,config.FACE_IDENTITY_THRESHOLD),1)
        cv2.putText(frame,
            f"DB    : {', '.join(self._db.keys()) or 'empty'}",
            (8,my+56),cv2.FONT_HERSHEY_SIMPLEX,0.54,(200,200,200),1)
        cv2.putText(frame,
            "e=Enroll  d=Delete  r=Relock",
            (8,my+76),cv2.FONT_HERSHEY_SIMPLEX,0.50,(130,130,130),1)
        return frame
