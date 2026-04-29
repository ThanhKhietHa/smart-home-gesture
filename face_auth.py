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
    base_options=python.BaseOptions(
        model_asset_path=config.FACE_MODEL_PATH,
        delegate=python.BaseOptions.Delegate.CPU  # Moves math to CUDA cores
    ),
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

_STABLE = [4,6,8,9,10,33,133,159,145,263,234,454,
           152,10,61,291,70,300,168,197]

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
    if my < fh:
        return typed, False, False
    ry = my - fh
    KH = 34
    for ri, row in enumerate(_KB_ROWS):
        n = len(row)
        kw = (fw-20)//10
        yt = 44 + ri*(KH+5)
        yb = yt+KH
        if not (yt <= ry <= yb):
            continue
        xoff = (fw - n*(kw+4))//2
        for ci, ch in enumerate(row):
            kx = xoff + ci*(kw+4)
            bw = kw*2 if ch in ("DEL","OK") else kw
            if kx <= mx <= kx+bw:
                if ch == "DEL":   return typed[:-1], False, False
                elif ch == "OK":  return typed, bool(typed.strip()), False
                else:             return typed+ch, False, False
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
    nose = lm[_NOSE].copy()
    eye_dist = np.linalg.norm(lm[_LEFT_EYE]-lm[_RIGHT_EYE]) + 1e-6
    return (lm - nose) / eye_dist

def _stable(lm_n):
    return lm_n[_STABLE]

def _head_pose(lm, shape):
    h, w = shape[:2]
    idx = [_NOSE,_CHIN,_LEFT_EYE,_RIGHT_EYE,_LEFT_MOUTH,_RIGHT_MOUTH]
    p2d = np.array([[lm[i,0]*w, lm[i,1]*h] for i in idx], dtype=np.float32)
    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float32)
    obj = np.array([[0,0,0],[0,-63.6,-12.5],[-43.3,32.7,-26],
                    [43.3,32.7,-26],[-28.9,-28.9,-24.1],[28.9,-28.9,-24.1]],
                   dtype=np.float32)
    _, rv, _ = cv2.solvePnP(obj, p2d, cam, np.zeros((4,1),np.float32))
    R, _ = cv2.Rodrigues(rv)
    yaw = float(np.degrees(np.arctan2(R[1,0],R[0,0])))
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
    if not db:
        return 'Unknown', 999.0, 999.0, False
    st = _stable(lm_n)
    best_n, best_se, best_ie = 'Unknown', 999.0, 999.0
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
# CLASS
# =====================================================================
class FaceAuth:

    def __init__(self):
        self._db = _load_db()
        self._state = _ST_RECOGNISE
        self._unlocked = False
        self._unlock_name = ""
        self._unlock_time = 0.0
        self._match_buf = deque(maxlen=config.FACE_CONFIRM_FRAMES)
        self._relock_buf = deque(maxlen=config.FACE_RELOCK_FRAMES)
        self._typed = ""

    def handle_key(self, key):
        if key == ord('e'):
            self._state = _ST_TYPING

    # ✅ FIXED FUNCTION
    def process_frame(self, frame, key, skip_inference=False):
        result = None
        lm = None
        face_ok = False

        if not skip_inference:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = _landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            lm = _to_np(result)

            if lm is not None:
                face_ok = True

        if self._state == _ST_RECOGNISE and face_ok:
            lm_n = _normalize(lm)
            name, se, ie, match = _identify(lm_n, self._db)
            if match:
                self._unlocked = True
                self._unlock_name = name

        return frame
