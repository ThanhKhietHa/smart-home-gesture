"""
face_auth.py — Face Recognition + Centroid Tracker (Sticky ID)
===============================================================
Fixes vs previous version:
  - GRACE_FRAMES = 20 (2 seconds at 10 FPS, was 2 = 200ms)
  - process_presence_only(): lightweight 5ms path for unlocked skip frames
    → called from face_thread every frame when unlocked+skipping
    → grace counter now increments correctly when face disappears
  - check_presence() kept for use inside _state_recognise (unlocked branch)
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
# TUNABLE CONSTANTS
# =====================================================================
GRACE_FRAMES  = 20          # frames before re-lock when face lost (~2s at 10 FPS)
IOU_THRESHOLD = 0.30        # IOU threshold for centroid tracker

# =====================================================================
# MEDIAPIPE — one landmarker at module level
# =====================================================================
_face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=getattr(config, 'FACE_DETECTION_CONFIDENCE', 0.35),
    min_face_presence_confidence=getattr(config, 'FACE_PRESENCE_CONFIDENCE', 0.35),
    output_facial_transformation_matrixes=False,
)
_landmarker = vision.FaceLandmarker.create_from_options(_face_options)

# =====================================================================
# LANDMARK INDICES
# =====================================================================
_NOSE        = 4
_LEFT_EYE    = 33
_RIGHT_EYE   = 263
_LEFT_MOUTH  = 61
_RIGHT_MOUTH = 291
_CHIN        = 152

_STABLE = [4, 6, 8, 9, 10, 33, 133, 159, 145, 263,
           234, 454, 152, 13, 61, 291, 70, 300, 168, 197]

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
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(aA + aB - inter + 1e-6)

    def update(self, box):
        if box is None:
            self._age += 1
            return self._id, False
        if self._box is None:
            self._box      = box
            self._centroid = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            self._id      += 1
            self._age      = 0
            return self._id, True
        iou = self._iou(self._box, box)
        self._box      = box
        self._centroid = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        self._age      = 0
        if iou >= IOU_THRESHOLD:
            return self._id, False
        else:
            self._id += 1
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
    return np.array([[l.x, l.y, l.z] for l in result.face_landmarks[0]],
                    dtype=np.float32)

def _get_box(result, shape):
    if not result.face_landmarks:
        return None
    H, W = shape[:2]
    xs = [l.x * W for l in result.face_landmarks[0]]
    ys = [l.y * H for l in result.face_landmarks[0]]
    return (int(min(xs)) - 10, int(min(ys)) - 10,
            int(max(xs)) + 10, int(max(ys)) + 10)

def _normalize(lm):
    nose     = lm[_NOSE].copy()
    eye_dist = np.linalg.norm(lm[_LEFT_EYE] - lm[_RIGHT_EYE]) + 1e-6
    return (lm - nose) / eye_dist

def _stable(lm_n):
    return lm_n[_STABLE]

def _head_pose(lm, shape):
    h, w = shape[:2]
    idx  = [_NOSE, _CHIN, _LEFT_EYE, _RIGHT_EYE, _LEFT_MOUTH, _RIGHT_MOUTH]
    p2d  = np.array([[lm[i, 0] * w, lm[i, 1] * h] for i in idx], dtype=np.float32)
    cam  = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
    obj  = np.array([[0, 0, 0], [0, -63.6, -12.5], [-43.3, 32.7, -26],
                     [43.3, 32.7, -26], [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1]],
                    dtype=np.float32)
    _, rv, _ = cv2.solvePnP(obj, p2d, cam, np.zeros((4, 1), np.float32))
    R, _     = cv2.Rodrigues(rv)
    yaw   = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    pitch = float(np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))))
    return yaw, pitch

def _shape_err(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))

def _cosine_err(a, b):
    af, bf = a.flatten(), b.flatten()
    return float(1.0 - np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-6))

# =====================================================================
# DATABASE
# =====================================================================
def _load_db():
    if not os.path.exists(config.ENROLLED_FILE):
        return {}
    try:
        with open(config.ENROLLED_FILE, 'rb') as f:
            db = pickle.load(f)
        print(f"[FACE] Loaded {len(db)} enrolled face(s): {list(db.keys())}")
        return db
    except Exception as e:
        print(f"[FACE] Could not load enrolled faces: {e}")
        return {}

def _save_db(db):
    with open(config.ENROLLED_FILE, 'wb') as f:
        pickle.dump(db, f)

def _identify(lm_n, db):
    if not db:
        return 'Unknown', 999.0, 999.0, False
    st     = _stable(lm_n)
    best_n = 'Unknown'
    best_se = 999.0
    best_ie = 999.0
    for name, data in db.items():
        se = _shape_err(st, data['stable'])
        ie = _cosine_err(st, data['stable'])
        if (0.5 * se + 0.5 * ie) < (0.5 * best_se + 0.5 * best_ie):
            best_n, best_se, best_ie = name, se, ie
    match = (best_se < config.FACE_SHAPE_THRESHOLD and
             best_ie < config.FACE_IDENTITY_THRESHOLD)
    if not match:
        best_n = 'Unknown'
    return best_n, best_se, best_ie, match

# =====================================================================
# KEYBOARD UI
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
    cv2.rectangle(panel, (10, 5), (w - 10, 38), (50, 50, 50), -1)
    cv2.putText(panel, typed + "|", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    cv2.putText(panel, "Type name then OK  |  ESC=cancel",
                (w - 340, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)
    KH = 34
    for ri, row in enumerate(_KB_ROWS):
        n    = len(row)
        kw   = (w - 20) // 10
        yt   = 44 + ri * (KH + 5)
        xoff = (w - n * (kw + 4)) // 2
        for ci, ch in enumerate(row):
            kx = xoff + ci * (kw + 4)
            bw = kw * 2 if ch in ("DEL", "OK") else kw
            bg = (0, 110, 0) if ch == "OK" else (110, 0, 0) if ch == "DEL" else (65, 65, 88)
            cv2.rectangle(panel, (kx, yt), (kx + bw, yt + KH), bg, -1)
            cv2.rectangle(panel, (kx, yt), (kx + bw, yt + KH), (130, 130, 130), 1)
            tx = kx + max((bw - len(ch) * 8) // 2, 2)
            cv2.putText(panel, ch, (tx, yt + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(panel, "Click keys  |  Enter=confirm  |  ESC=cancel",
                (10, _KB_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)
    return np.vstack([frame, panel])

def _kb_hittest(fh, fw, mx, my, typed):
    if my < fh:
        return typed, False, False
    ry = my - fh
    KH = 34
    for ri, row in enumerate(_KB_ROWS):
        n    = len(row)
        kw   = (fw - 20) // 10
        yt   = 44 + ri * (KH + 5)
        if not (yt <= ry <= yt + KH):
            continue
        xoff = (fw - n * (kw + 4)) // 2
        for ci, ch in enumerate(row):
            kx = xoff + ci * (kw + 4)
            bw = kw * 2 if ch in ("DEL", "OK") else kw
            if kx <= mx <= kx + bw:
                if ch == "DEL":  return typed[:-1], False, False
                elif ch == "OK": return typed, bool(typed.strip()), False
                else:            return typed + ch,  False,         False
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
        self._clicked      = False
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
            self._state   = _ST_TYPING
            self._typed   = ""
            self._clicked = False
        elif key == ord('d'):
            if self._db:
                self._state     = _ST_DELETE
                self._del_names = list(self._db.keys())
        elif key == ord('r'):
            self._force_relock("Manual relock")

    # =================================================================
    # process_presence_only — ~5ms, called every skipped frame when unlocked
    # This is the fix for the grace period bug:
    #   face_thread was writing raw frame directly and skipping entirely,
    #   so grace_count never incremented when face disappeared.
    #   Now face_thread calls this instead — runs detector, updates tracker,
    #   increments grace, triggers relock if face gone long enough.
    # =================================================================
    def process_presence_only(self, frame):
        """
        Lightweight presence check (~5-10ms).
        Runs MediaPipe face detector to get bounding box only —
        no landmark extraction, no recognition math.
        Updates centroid tracker and grace counter every frame.
        Returns annotated frame.
        """
        try:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = _landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            box = _get_box(result, frame.shape)
        except Exception:
            box = None

        return self.check_presence(frame, box)

    # =================================================================
    # check_presence — updates tracker+grace given a box (may be None)
    # =================================================================
    def check_presence(self, frame, box):
        """
        Given a bounding box (or None), update tracker and grace counter.
        Triggers relock if grace exceeds GRACE_FRAMES.
        Returns annotated frame.
        """
        if box is not None:
            track_id, _ = self._tracker.update(box)
            self._grace_count  = 0
            self._track_status = f"UNLOCKED tracking id={track_id}"
            # Draw green box
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)
            cv2.putText(frame, self._unlock_name,
                        (x1, max(y1 - 10, 85)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            self._tracker.lost()
            self._grace_count += 1
            self._track_status = f"LOST grace={self._grace_count}/{GRACE_FRAMES}"
            if self._unlocked and self._grace_count >= GRACE_FRAMES:
                self._force_relock("Face lost — grace period expired")
        return frame

    # =================================================================
    # process_frame — full processing (recognition + drawing)
    # =================================================================
    def process_frame(self, frame, key):
        if self._state == _ST_TYPING:
            return self._state_typing(frame, key)
        if self._state == _ST_DELETE:
            return self._state_delete(frame, key)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        lm  = _to_np(result)
        box = _get_box(result, frame.shape)

        if self._state == _ST_ENROLLING:
            return self._state_enrolling(frame, result, lm, key)

        return self._state_recognise(frame, result, lm, box, key)

    def _state_recognise(self, frame, result, lm, box, key):
        self._frame_counter += 1

        if not self._unlocked:
            # LOCKED: run recognition every N frames
            process_this = (self._frame_counter %
                            getattr(config, 'FACE_PROCESS_EVERY_N_FRAMES_LOCKED', 2) == 0)
            if process_this and box is not None:
                track_id, is_new = self._tracker.update(box)
                self._grace_count = 0
                H = frame.shape[0]
                if result and result.face_landmarks:
                    ys = [pt.y * H for pt in result.face_landmarks[0]]
                    face_ok = (max(ys) - min(ys)) / H >= config.FACE_MIN_HEIGHT_FRAC
                else:
                    face_ok = bool(self._tracker.box)
                if face_ok and lm is not None:
                    self._run_recognition(lm, frame.shape, track_id)
            else:
                if box is not None:
                    self._tracker.update(box)
                    self._grace_count = 0
                else:
                    self._tracker.lost()
                    self._grace_count += 1

            # Draw locked box
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 60, 220), 2)
                name_label = self._last_cand if self._last_cand != "—" else "Scanning..."
                cv2.putText(frame, name_label, (x1, max(y1 - 10, 85)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 100, 255), 2)

            # Red border when locked
            cv2.rectangle(frame, (0, 0),
                          (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 200), 4)

        else:
            # UNLOCKED: lightweight presence check (recognition already skipped
            # at the face_thread level; this handles the non-skip frames)
            return self.check_presence(frame, box)

        # Auth timeout
        if (self._unlocked and
                (time.time() - self._unlock_time) > config.FACE_AUTH_TIMEOUT):
            self._force_relock("Session timeout")

        return frame

    def _run_recognition(self, lm, shape, track_id):
        try:
            yaw, pitch = _head_pose(lm, shape)
        except Exception:
            return
        if abs(yaw) > 55 or abs(pitch) > 45:
            return
        lm_n            = _normalize(lm)
        name, se, ie, match = _identify(lm_n, self._db)
        self._last_se   = se
        self._last_ie   = ie
        self._last_cand = name
        self._match_buf.append(match)
        if (match and sum(self._match_buf) >= int(config.FACE_CONFIRM_FRAMES * 0.7)
                and not self._unlocked):
            self._unlocked    = True
            self._unlock_name = name
            self._unlock_time = time.time()
            self._tracked_id  = track_id
            print(f"[FACE] UNLOCKED — {name}")

    def _force_relock(self, reason=""):
        if reason:
            print(f"[FACE] Re-locked: {reason}")
        self._unlocked    = False
        self._unlock_name = ""
        self._tracked_id  = -1
        self._grace_count = 0
        self._match_buf.clear()
        self._tracker.reset()

    # =================================================================
    # TYPING / ENROLL / DELETE STATES (unchanged)
    # =================================================================
    def _state_typing(self, frame, key):
        combined = _draw_keyboard(frame, self._typed)
        fh, fw   = frame.shape[:2]
        if self._clicked:
            self._typed, done, _ = _kb_hittest(fh, fw, self._mx, self._my, self._typed)
            self._clicked = False
            if done and self._typed.strip():
                self._start_enroll(self._typed.strip())
        if key == 27:
            self._state = _ST_RECOGNISE
        elif key in (13, 10):
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

    def _state_enrolling(self, frame, result, lm, key):
        name      = self._enroll_name
        collected = self._enroll_col
        TIMEOUT   = 50.0
        remaining = TIMEOUT - (time.time() - self._enroll_start)
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
                        cv2.circle(frame, (int(pt.x * W), int(pt.y * H)), 2, (0, 255, 0), -1)
                else:
                    status = f"Turn straight yaw={yaw:.0f} pitch={pitch:.0f}"
                    H, W = frame.shape[:2]
                    for pt in result.face_landmarks[0]:
                        cv2.circle(frame, (int(pt.x * W), int(pt.y * H)), 2, (0, 140, 255), -1)
            except Exception:
                status = "Adjust position"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 115), (0, 55, 150), -1)
        cv2.putText(frame, f"ENROLLING: {name}",
                    (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(frame, status,
                    (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 100) if is_good else (0, 150, 255), 2)
        prog = len(collected) / config.FACE_ENROLL_TARGET
        bw   = frame.shape[1] - 40
        cv2.rectangle(frame, (20, 118), (20 + bw, 140), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 118), (20 + int(bw * prog), 140), (0, 220, 100), -1)
        cv2.putText(frame,
                    f"{int(prog*100)}% — 30-50cm, look straight | {remaining:.0f}s left",
                    (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "ESC = cancel",
                    (20, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if key == 27:
            print("[FACE] Enrollment cancelled.")
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
            print(f"[FACE] '{name}' enrolled ({len(collected)} frames). Photo: {photo}")
        else:
            print(f"[FACE] Enroll failed — only {len(collected)} frames.")
        self._state = _ST_RECOGNISE
        self._force_relock()

    def _state_delete(self, frame, key):
        panel = np.full_like(frame, 25)
        cv2.putText(panel, "DELETE WHICH FACE?",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
        cv2.putText(panel, "Press number key | ESC = cancel",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)
        for i, nm in enumerate(self._del_names[:5]):
            cv2.putText(panel, f"  {i+1}. {nm}",
                        (20, 145 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 170, 0), 2)
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

    # =================================================================
    # HUD DRAWING
    # =================================================================
    def draw_status_bar(self, frame):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 68),
                      (0, 100, 0) if self._unlocked else (0, 0, 120), -1)
        if self._unlocked:
            rem = max(0.0, config.FACE_AUTH_TIMEOUT - (time.time() - self._unlock_time))
            cv2.putText(frame, f"UNLOCKED [{self._unlock_name}]",
                        (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 80), 2)
            cv2.putText(frame, f"Session: {int(rem)}s",
                        (frame.shape[1] - 160, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 255, 160), 1)
            if self._grace_count > 0:
                cv2.putText(frame, f"GRACE {self._grace_count}/{GRACE_FRAMES}",
                            (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        else:
            if not self._db:
                msg = "NO FACES — press 'e'"
            elif self._tracker.box is None:
                msg = "LOCKED — No face"
            else:
                prog = sum(self._match_buf) / max(config.FACE_CONFIRM_FRAMES, 1)
                msg  = f"LOCKED — {int(prog * 100)}%"
            cv2.putText(frame, msg, (12, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 140, 255), 2)
        return frame

    def draw_debug(self, frame):
        H, W = frame.shape[:2]
        mc   = lambda v, t: (0, 220, 0) if v < t else (0, 80, 220)
        lines = [
            (f"S:{self._last_se:.3f}/{config.FACE_SHAPE_THRESHOLD}",
             mc(self._last_se, config.FACE_SHAPE_THRESHOLD)),
            (f"C:{self._last_ie:.3f}/{config.FACE_IDENTITY_THRESHOLD}",
             mc(self._last_ie, config.FACE_IDENTITY_THRESHOLD)),
            (f"{self._track_status[:18]}", (180, 180, 180)),
        ]
        panel_w = 185
        panel_h = len(lines) * 15 + 6
        x0 = W - panel_w - 4
        y0 = 72
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (W - 2, y0 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (x0 + 5, y0 + 12 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return frame
