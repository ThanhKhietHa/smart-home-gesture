"""
face_auth.py — Stable & Less Glitchy Face Recognition
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

# ==================== MEDIAPIPE SETUP ====================
_face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.35,
    min_face_presence_confidence=0.35,
    output_facial_transformation_matrixes=True,
)
_landmarker = vision.FaceLandmarker.create_from_options(_face_options)

# ==================== HELPERS ====================
def _to_np(result):
    if not result.face_landmarks:
        return None
    return np.array([[l.x, l.y, l.z] for l in result.face_landmarks[0]], dtype=np.float32)

def _normalize(lm):
    nose = lm[4]
    eye_dist = np.linalg.norm(lm[33] - lm[263]) + 1e-6
    return (lm - nose) / eye_dist

def _stable(lm_n):
    idx = [4, 33, 263, 152, 61, 291, 10, 168]
    return lm_n[idx]

def _shape_err(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))

def _cosine_err(a, b):
    af = a.flatten()
    bf = b.flatten()
    return float(1.0 - np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-6))

# ==================== FACE AUTH CLASS ====================
class FaceAuth:
    def __init__(self):
        self._db = self._load_db()

        # Auth state
        self._unlocked = False
        self._unlock_name = ""

        # Anti-glitch buffers
        self._match_buffer = deque(maxlen=10)      # more stable unlock
        self._no_match_buffer = deque(maxlen=15)   # slower to relock

        # Debug
        self.last_se = 0.0
        self.last_ie = 0.0
        self.last_name = "—"

        # Enroll state (simple version)
        self._enrolling = False
        self._enroll_name = ""
        self._enroll_frames = []

    def _load_db(self):
        if not os.path.exists(config.ENROLLED_FILE):
            return {}
        try:
            with open(config.ENROLLED_FILE, 'rb') as f:
                db = pickle.load(f)
            print(f"[FACE] Loaded {len(db)} enrolled face(s)")
            return db
        except Exception as e:
            print(f"[FACE] Load error: {e}")
            return {}

    def is_unlocked(self):
        return self._unlocked

    def unlocked_name(self):
        return self._unlock_name

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        lm = _to_np(result)

        matched = False
        name = "Unknown"

        if lm is not None:
            lm_n = _normalize(lm)
            stable = _stable(lm_n)

            best_score = 999.0
            best_name = "Unknown"

            for n, data in self._db.items():
                score = 0.6 * _shape_err(stable, data.get('stable', stable)) + \
                        0.4 * _cosine_err(stable, data.get('stable', stable))
                if score < best_score:
                    best_score = score
                    best_name = n

            self.last_se = _shape_err(stable, self._db.get(best_name, {}).get('stable', stable))
            self.last_ie = _cosine_err(stable, self._db.get(best_name, {}).get('stable', stable))
            self.last_name = best_name

            matched = (self.last_se < config.FACE_SHAPE_THRESHOLD and 
                       self.last_ie < config.FACE_IDENTITY_THRESHOLD)

        # === Smoothing to reduce glitch ===
        self._match_buffer.append(matched)
        self._no_match_buffer.append(not matched)

        # Unlock condition
        if not self._unlocked and sum(self._match_buffer) >= 7:
            self._unlocked = True
            self._unlock_name = self.last_name
            print(f"✅ UNLOCKED — {self._unlock_name}")

        # Relock condition (more tolerant)
        if self._unlocked and sum(self._no_match_buffer) >= 12:
            self._unlocked = False
            print("🔒 Re-locked (face lost)")

        # Draw UI
        self._draw_ui(frame, lm, matched)

        return frame

    def _draw_ui(self, frame, lm, matched):
        if lm is not None:
            H, W = frame.shape[:2]
            color = (0, 255, 0) if matched else (0, 100, 255)
            for pt in lm:
                x = int(pt[0] * W)
                y = int(pt[1] * H)
                cv2.circle(frame, (x, y), 2, color, -1)

        # Top status bar - big and clear
        if self._unlocked:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (0, 100, 0), -1)
            cv2.putText(frame, f"UNLOCKED — {self._unlock_name}", (25, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 255, 80), 3)
        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (0, 0, 100), -1)
            cv2.putText(frame, "LOCKED — Show your enrolled face", (25, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 180, 255), 2)

        # Bottom debug info
        cv2.putText(frame, f"Shape: {self.last_se:.3f}   Cos: {self.last_ie:.3f}", 
                    (15, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, f"Detected: {self.last_name}", 
                    (15, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    def handle_key(self, key):
        if key == ord('r'):
            self._unlocked = False
            self._match_buffer.clear()
            self._no_match_buffer.clear()
            print("[FACE] Manually re-locked")
        elif key == ord('e'):
            print("[FACE] Enroll mode - Not implemented in this stable version yet")

# For compatibility with main.py
    def draw_status_bar(self, frame):
        return frame

    def draw_debug(self, frame):
        return frame
