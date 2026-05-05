"""
face_auth.py — Face Recognition + Centroid Tracker (Sticky ID)
==============================================================

Architecture — 3 layers:

  Layer 1  DETECT   (every frame, lightweight)
           MediaPipe finds face bounding box + landmarks.
           Cost: ~2ms — always runs.

  Layer 2  TRACK    (every frame, near-zero cost)
           CentroidTracker matches box to known track by IOU.
           If match found → update position, skip recognition.
           If new/unmatched face → trigger Layer 3.

  Layer 3  IDENTIFY (only on new face OR every N frames, expensive)
           Full landmark comparison against enrolled DB.
           Cost: ~40ms — runs as rarely as possible.

Grace period:
  When tracked face disappears, a countdown starts.
  If face reappears within GRACE_FRAMES → stay unlocked (no re-scan).
  If countdown expires → re-lock and reset tracker.

Periodic re-verify:
  Even while tracked, re-runs identification every REVERIFY_FRAMES.
  Catches face-swap attacks (person walks away, different person steps in).
  If re-verify fails → immediate re-lock.

State machine (unchanged from previous version):
  RECOGNISE → TYPING → ENROLLING → RECOGNISE
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
# TUNABLE TRACKER CONSTANTS
# (exposed here so they're easy to find — could move to config.py)
# =====================================================================
GRACE_FRAMES    = 40    # ~2s at 20fps — frames before re-locking when face lost
IOU_THRESHOLD   = 0.30  # minimum IOU to consider same face across frames

# NOTE: Periodic re-verify is DISABLED per user preference.
# Recognition only runs when:
#   1. Face first enters frame (new track, not yet unlocked)
#   2. System is locked and scanning for a match
# Once unlocked, tracker holds identity until face is fully lost.

# =====================================================================
# MEDIAPIPE
# =====================================================================
_face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=config.FACE_MODEL_PATH,
        delegate=python.BaseOptions.Delegate.GPU
    ),,
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
_NOSE        = 4
_LEFT_EYE    = 33
_RIGHT_EYE   = 263
_LEFT_MOUTH  = 61
_RIGHT_MOUTH = 291
_CHIN        = 152

# Stable structural landmarks — no duplicate indices
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
    """
    Lightweight single-face centroid tracker.
    Uses IOU (Intersection over Union) to match bounding boxes
    across frames — much faster than any ML-based tracker.

    Tracks ONE face only (matching system requirements).
    """

    def __init__(self):
        self._box      = None    # (x1,y1,x2,y2) of tracked face
        self._centroid = None    # (cx,cy) centre of tracked box
        self._id       = 0       # increments each time a NEW face is detected
        self._age      = 0       # frames since this track was last matched

    @staticmethod
    def _iou(boxA, boxB):
        """Intersection over Union between two (x1,y1,x2,y2) boxes."""
        xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        if inter == 0:
            return 0.0
        aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / float(aA + aB - inter + 1e-6)

    def update(self, box):
        """
        Call with (x1,y1,x2,y2) bounding box each frame a face is found.
        Returns (track_id, is_new_face).
          is_new_face=True  → recognition should run
          is_new_face=False → face already tracked, skip recognition
        """
        if box is None:
            self._age += 1
            return self._id, False

        if self._box is None:
            # First detection ever
            self._box      = box
            self._centroid = ((box[0]+box[2])//2, (box[1]+box[3])//2)
            self._id      += 1
            self._age      = 0
            return self._id, True

        iou = self._iou(self._box, box)

        if iou >= IOU_THRESHOLD:
            # Same face — update position
            self._box      = box
            self._centroid = ((box[0]+box[2])//2, (box[1]+box[3])//2)
            self._age      = 0
            return self._id, False
        else:
            # Different face (moved too much or completely new person)
            self._box      = box
            self._centroid = ((box[0]+box[2])//2, (box[1]+box[3])//2)
            self._id      += 1
            self._age      = 0
            return self._id, True

    def lost(self):
        """Call every frame when NO face is detected."""
        self._age += 1

    def reset(self):
        """Full tracker reset — clears all state."""
        self._box      = None
        self._centroid = None
        self._age      = 0

    @property
    def age(self):
        return self._age

    @property
    def centroid(self):
        return self._centroid

    @property
    def box(self):
        return self._box

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
            kx   = xoff + ci*(kw+4)
            bw   = kw*2 if ch in ("DEL","OK") else kw
            bg   = (0,110,0) if ch=="OK" else (110,0,0) if ch=="DEL" else (65,65,88)
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
# LANDMARK HELPERS
# =====================================================================
def _to_np(result):
    if not result.face_landmarks:
        return None
    return np.array([[l.x,l.y,l.z] for l in result.face_landmarks[0]],
                    dtype=np.float32)

def _get_box(result, shape):
    """
    Compute bounding box (x1,y1,x2,y2) in pixels from face landmarks.
    Returns None if no face detected.
    """
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
# FACEAUTH CLASS
# =====================================================================
class FaceAuth:
    """
    Public API (used by main.py — unchanged):
      process_frame(frame, key) → annotated frame
      handle_key(key)
      is_unlocked() → bool
      unlocked_name() → str
      enrolled_names() → list
      draw_status_bar(frame)
      draw_debug(frame)
      mouse_callback(event, x, y, flags, param)
    """

    def __init__(self):
        self._db           = _load_db()
        self._state        = _ST_RECOGNISE

        # ── Auth state ────────────────────────────────────────────────
        self._unlocked     = False
        self._unlock_name  = ""
        self._unlock_time  = 0.0

        # ── Centroid tracker ──────────────────────────────────────────
        self._tracker      = _CentroidTracker()
        self._grace_count  = 0      # counts up when face is lost
        self._tracked_id   = -1     # track ID of the currently unlocked face

        # ── Initial unlock buffer (still needed for first identification)
        self._match_buf    = deque(maxlen=config.FACE_CONFIRM_FRAMES)

        # ── Debug metrics ─────────────────────────────────────────────
        self.last_se       = 999.0
        self.last_ie       = 999.0
        self.last_cand     = "—"
        self._track_status = "SCANNING"  # shown in debug panel

        # ── Enroll state ──────────────────────────────────────────────
        self._enroll_name  = ""
        self._enroll_col   = []
        self._enroll_start = 0.0
        self._del_names    = []
        self._typed        = ""

        # ── Mouse ─────────────────────────────────────────────────────
        self._mx = 0;  self._my = 0;  self._clicked = False

    # ── Mouse callback ────────────────────────────────────────────────
    def mouse_callback(self, event, x, y, flags, param):
        self._mx, self._my = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked = True

    # ── Public queries ────────────────────────────────────────────────
    def is_unlocked(self)     -> bool:  return self._unlocked
    def unlocked_name(self)   -> str:   return self._unlock_name
    def enrolled_names(self)  -> list:  return list(self._db.keys())

    # ── Key handler ───────────────────────────────────────────────────
    def handle_key(self, key):
        if self._state != _ST_RECOGNISE:
            return
        if key == ord('e'):
            self._state = _ST_TYPING;  self._typed = "";  self._clicked = False
        elif key == ord('d'):
            if self._db:
                self._state = _ST_DELETE;  self._del_names = list(self._db.keys())
        elif key == ord('r'):
            self._force_relock("Manual relock")

    # ── Main per-frame entry ──────────────────────────────────────────
    def process_frame(self, frame, key):
        # Fast path — no detection for UI-only states
        if self._state == _ST_TYPING:
            return self._state_typing(frame, key)
        if self._state == _ST_DELETE:
            return self._state_delete(frame, key)

        # Run MediaPipe detection (every frame — lightweight bbox only)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        lm  = _to_np(result)
        box = _get_box(result, frame.shape)

        if self._state == _ST_ENROLLING:
            return self._state_enrolling(frame, result, lm, key)

        # ── RECOGNISE state with Sticky ID tracker ────────────────────
        return self._state_recognise(frame, result, lm, box, key)

    # ── RECOGNISE STATE with centroid tracker ─────────────────────────
    def _state_recognise(self, frame, result, lm, box, key):

        if box is not None:
            # ── Face is present — update tracker ──────────────────────
            track_id, is_new = self._tracker.update(box)
            self._grace_count = 0   # reset grace period

            # Face size guard
            H = frame.shape[0]
            ys = [pt.y*H for pt in result.face_landmarks[0]]
            face_ok = (max(ys)-min(ys))/H >= config.FACE_MIN_HEIGHT_FRAC

            # Decide whether to run recognition this frame:
            #   - System locked AND face is present → always try to identify
            #   - System unlocked → NEVER re-verify (trust tracker fully)
            #   - New face while unlocked → IGNORE (stay unlocked until
            #     current face is fully lost, THEN scan the new one)
            run_recognition = (not self._unlocked) and face_ok and (lm is not None)

            if run_recognition:
                self._run_recognition(lm, frame.shape, track_id)

            # ── Draw face box ──────────────────────────────────────────
            x1, y1, x2, y2 = box
            if self._unlocked:
                # Always show green when unlocked — regardless of who is there
                box_color  = (0, 220, 0)
                thickness  = 3
                name_label = self._unlock_name
                label_col  = (0, 255, 0)
            else:
                box_color  = (0, 60, 220)
                thickness  = 2
                name_label = self.last_cand if self.last_cand != "—" else "Scanning..."
                label_col  = (60, 100, 255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, thickness)
            cv2.putText(frame, name_label, (x1, max(y1-10, 85)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_col, 2)

            # Draw landmarks
            H2, W2 = frame.shape[:2]
            for pt in result.face_landmarks[0]:
                cv2.circle(frame,
                           (int(pt.x*W2), int(pt.y*H2)),
                           1, box_color, -1)

            self._track_status = (
                f"UNLOCKED — tracking  id={track_id}" if self._unlocked
                else f"SCANNING  id={track_id}"
            )

        else:
            # ── No face detected this frame ───────────────────────────
            self._tracker.lost()
            self._grace_count += 1
            self._track_status = f"LOST  grace={self._grace_count}/{GRACE_FRAMES}"

            if self._unlocked:
                if self._grace_count >= GRACE_FRAMES:
                    self._force_relock("Face lost — grace period expired")
                # else stay unlocked during grace period

        # Auth timeout
        if (self._unlocked
                and (time.time()-self._unlock_time) > config.FACE_AUTH_TIMEOUT):
            self._force_relock("Session timeout")

        # Red border when locked
        if not self._unlocked:
            cv2.rectangle(frame,
                          (0,0), (frame.shape[1]-1, frame.shape[0]-1),
                          (0,0,200), 5)

        return frame

    # ── Run full recognition and update auth state ────────────────────
    def _run_recognition(self, lm, shape, track_id):
        """
        Runs landmark comparison against DB.
        Only called when system is LOCKED — never when already unlocked.
        """
        try:
            yaw, pitch = _head_pose(lm, shape)
        except Exception:
            return

        if abs(yaw) > 55 or abs(pitch) > 45:
            return

        lm_n = _normalize(lm)
        name, se, ie, match = _identify(lm_n, self._db)

        self.last_se   = se
        self.last_ie   = ie
        self.last_cand = name

        self._match_buf.append(match)

        # Enough consecutive matches → unlock
        if (match
                and sum(self._match_buf) >= int(config.FACE_CONFIRM_FRAMES * 0.7)
                and not self._unlocked):
            self._unlocked    = True
            self._unlock_name = name
            self._unlock_time = time.time()
            self._tracked_id  = track_id
            print(f"[FACE] UNLOCKED — {name}  (track_id={track_id})")

    def _force_relock(self, reason=""):
        if reason:
            print(f"[FACE] Re-locked: {reason}")
        self._unlocked    = False
        self._unlock_name = ""
        self._tracked_id  = -1
        self._grace_count = 0
        self._match_buf.clear()
        self._tracker.reset()

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
            print("  Tips: 30-50cm, good lighting, face straight.")
        self._state = _ST_RECOGNISE
        self._force_relock()

    # ── STATE: DELETE ─────────────────────────────────────────────────
    def _state_delete(self, frame, key):
        panel = np.full_like(frame, 25)
        cv2.putText(panel,"DELETE WHICH FACE?",(20,55),
                    cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,100,255),2)
        cv2.putText(panel,"Press number key  |  ESC = cancel",
                    (20,95),cv2.FONT_HERSHEY_SIMPLEX,0.7,(150,150,150),1)
        for i, nm in enumerate(self._del_names):
            cv2.putText(panel,f"  {i+1}.  {nm}",
                        (20,145+i*45),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,170,0),2)
        if key == 27:
            self._state = _ST_RECOGNISE
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(self._del_names):
                nm = self._del_names[idx]
                del self._db[nm];  _save_db(self._db)
                print(f"[FACE] Deleted: {nm}")
            self._state = _ST_RECOGNISE
            self._force_relock()
        return panel

    # ── Draw auth status bar ──────────────────────────────────────────
    def draw_status_bar(self, frame):
        cv2.rectangle(frame,(0,0),(frame.shape[1],72),
                      (0,100,0) if self._unlocked else (0,0,120),-1)
        if self._unlocked:
            rem = max(0.0, config.FACE_AUTH_TIMEOUT-(time.time()-self._unlock_time))
            cv2.putText(frame,f"UNLOCKED  [ {self._unlock_name} ]",
                        (15,48),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,80),2)
            cv2.putText(frame,f"Session: {int(rem)}s",
                        (frame.shape[1]-170,48),
                        cv2.FONT_HERSHEY_SIMPLEX,0.75,(160,255,160),1)
            # Grace period warning — shows when face is briefly lost
            if self._grace_count > 0:
                grace_pct = self._grace_count / GRACE_FRAMES
                cv2.putText(frame,
                    f"TRACKING... grace {self._grace_count}/{GRACE_FRAMES}",
                    (15,68),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),1)
        else:
            if not self._db:
                msg = "NO FACES ENROLLED — press 'e'"
            elif self._tracker.box is None:
                msg = "LOCKED — No face in view"
            else:
                prog = sum(self._match_buf)/max(config.FACE_CONFIRM_FRAMES,1)
                msg  = f"LOCKED — Scanning... {int(prog*100)}%"
            cv2.putText(frame,msg,(15,48),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(80,140,255),2)
            if self._match_buf:
                prog = sum(self._match_buf)/config.FACE_CONFIRM_FRAMES
                bw   = frame.shape[1]-40
                cv2.rectangle(frame,(20,76),(20+bw,92),(50,50,50),-1)
                cv2.rectangle(frame,(20,76),
                    (20+int(bw*min(prog,1.0)),92),(0,180,255),-1)
        return frame

    # ── Draw debug panel ──────────────────────────────────────────────
    def draw_debug(self, frame):
        """
        Compact debug overlay — top-right corner, semi-transparent,
        does NOT block the camera view.
        """
        H, W = frame.shape[:2]
        mc = lambda v, t: (0, 220, 0) if v < t else (0, 80, 220)

        lines = [
            (f"Shape : {self.last_se:.3f} / {config.FACE_SHAPE_THRESHOLD}",
             mc(self.last_se, config.FACE_SHAPE_THRESHOLD)),
            (f"Cosine: {self.last_ie:.3f} / {config.FACE_IDENTITY_THRESHOLD}",
             mc(self.last_ie, config.FACE_IDENTITY_THRESHOLD)),
            (f"Track : {self._track_status}", (180, 180, 180)),
            (f"Grace : {self._grace_count}/{GRACE_FRAMES}", (140, 140, 140)),
            ("e=Enroll  d=Delete  r=Relock", (100, 100, 100)),
        ]

        # Semi-transparent background — only as wide as needed
        panel_w = 230
        panel_h = len(lines) * 18 + 10
        x0 = W - panel_w - 4
        y0 = 78   # just below the status bar

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (W - 2, y0 + panel_h),
                      (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text,
                        (x0 + 5, y0 + 14 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        return frame
