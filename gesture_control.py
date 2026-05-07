"""
gesture_control.py — Hand Gesture Recognition & Device Control
New: Window mode entered via Thumb Up
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import numpy as np
from collections import deque
import config

# =====================================================================
# MEDIAPIPE
# =====================================================================
_hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.HAND_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
)
_landmarker = vision.HandLandmarker.create_from_options(_hand_options)

_GS_IDLE    = "IDLE"
_GS_CONFIRM = "CONFIRM"
_GS_WINDOW  = "WINDOW"  # New state for window control mode

# =====================================================================
# OPTIMIZATION: Set of valid gestures for quick lookup
# =====================================================================
_VALID_GESTURES = set(config.GESTURE_COMMANDS.keys())

# =====================================================================
# GEOMETRY
# =====================================================================
def _dist(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

# =====================================================================
# GESTURE DETECTION
# =====================================================================
def detect_gesture(lm):
    """
    Returns gesture name string.
    """
    try:
        if not lm or len(lm) < 21:
            return "No hand"

        wrist      = lm[0]
        thumb_tip  = lm[4];  thumb_cmc  = lm[1]
        index_tip  = lm[8];  index_mcp  = lm[5]
        middle_tip = lm[12]; middle_mcp = lm[9]
        ring_tip   = lm[16]; ring_mcp   = lm[13]
        pinky_tip  = lm[20]; pinky_mcp  = lm[17]

        def ext(tip, mcp, thr=0.07):
            return tip.y < mcp.y - thr

        ie = ext(index_tip,  index_mcp)
        me = ext(middle_tip, middle_mcp)
        re = ext(ring_tip,   ring_mcp)
        pe = ext(pinky_tip,  pinky_mcp)
        n  = sum([ie, me, re, pe])

        tp = _dist(thumb_tip, wrist)
        tv = thumb_tip.y - thumb_cmc.y

        # =============================================================
        # POINTING UP
        # =============================================================
        if ie and not me and not re and not pe:
            v = index_tip.y - index_mcp.y
            if v < -0.12:
                return "Pointing Up"

        # =============================================================
        # PEACE SIGN
        # =============================================================
        if ie and me and not re and not pe:      
            return "Peace Sign"

        # =============================================================
        # THUMB GESTURES (fist with thumb)
        # =============================================================
        fingers_curled = (not ie and not me and not re and not pe)
        
        if fingers_curled and tp > 0.18:
            if tv < -0.10:
                return "Thumb Up"
            if tv > 0.05:
                return "Thumb Down"

        # =============================================================
        # OPEN PALM
        # =============================================================
        if n >= 3:
            return "Open Palm"

        # =============================================================
        # FIST
        # =============================================================
        if fingers_curled and tp < 0.22:
            return "Fist"

        return "Unknown"

    except Exception:
        return "No hand"

# =====================================================================
# GESTURE CONTROLLER
# =====================================================================
class GestureControl:

    def __init__(self):
        self._state        = _GS_IDLE
        self._pending      = None
        self._hold_start   = 0.0
        self._cur_gesture  = None
        self._conf_gesture = None
        self._conf_start   = 0.0
        self._conf_entry   = 0.0
        self._buf          = deque(maxlen=6)
        self.device_states = dict(config.DEVICE_INITIAL_STATES)
        self.window_mode   = False  # Track if we're in window control mode
        
        # OPTIMIZATION: Pre-filter valid gestures for faster checking
        self._valid_gestures = set(config.GESTURE_COMMANDS.keys())
        print(f"[GESTURE] Active gestures: {list(self._valid_gestures)}")

    # ── Main entry point ──────────────────────────────────────────────
    def process_frame(self, frame, mqtt, face_unlocked):
        """
        Returns (annotated_frame, feedback_or_None).
        """
        try:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = _landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            lm = result.hand_landmarks[0] if result.hand_landmarks else None
        except Exception:
            lm = None

        # Safe gesture detection
        self._buf.append(detect_gesture(lm))
        detected = self._smooth()

        # OPTIMIZATION: Early reject for invalid gestures (saves UI drawing)
        is_valid = detected in self._valid_gestures

        # Draw hand landmarks
        if lm is not None:
            try:
                H, W = frame.shape[:2]
                color = (0, 255, 0) if is_valid else (100, 100, 100)
                for pt in lm:
                    cx = int(max(0, min(pt.x * W, W-1)))
                    cy = int(max(0, min(pt.y * H, H-1)))
                    cv2.circle(frame, (cx, cy), 3, color, -1)
            except Exception:
                pass

        # Blocked — face not authenticated
        if not face_unlocked:
            self._reset()
            cv2.putText(frame, "FACE AUTH REQUIRED",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 220), 2)
            self._draw_devices(frame)
            return frame, None

        # Show window mode indicator
        if self.window_mode:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 100, 100), -1)
            cv2.putText(frame, "WINDOW CONTROL MODE - Thumb Up=Open  Thumb Down=Close  Open Palm=Exit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Route to state
        feedback = None
        if self._state == _GS_CONFIRM:
            feedback = self._do_confirm(frame, detected, mqtt)
        elif self.window_mode:
            # Handle window mode separately
            feedback = self._do_window_mode(frame, detected, mqtt)
        else:
            # Only show detection UI for valid gestures or during confirm
            if is_valid or self._pending is not None:
                self._do_detection(frame, detected)
            else:
                cv2.putText(frame, f"Gesture: {detected} (not mapped)",
                            (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (100, 100, 100), 1)

        self._draw_devices(frame)
        return frame, feedback

    # ── Window Mode Handler ───────────────────────────────────────────
    def _do_window_mode(self, frame, detected, mqtt):
        """Handle gestures in window control mode"""
        
        cv2.putText(frame, "WINDOW MODE ACTIVE",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Open Palm = Exit window mode
        if detected == "Open Palm":
            self.window_mode = False
            self._reset()
            return ("Window mode EXITED", (0, 255, 255))
        
        # Thumb Up = Open window
        if detected == "Thumb Up":
            mqtt.publish("window", "open")
            self.device_states["window"] = "open"
            self.window_mode = False  # Exit after action
            self._reset()
            return ("Window OPENED", (0, 255, 0))
        
        # Thumb Down = Close window
        if detected == "Thumb Down":
            mqtt.publish("window", "close")
            self.device_states["window"] = "closed"
            self.window_mode = False  # Exit after action
            self._reset()
            return ("Window CLOSED", (0, 0, 255))
        
        # Show instructions
        cv2.putText(frame, "Thumb UP = OPEN WINDOW",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Thumb DOWN = CLOSE WINDOW",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Open Palm = EXIT",
                    (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return None

    # ── Enter Window Mode ─────────────────────────────────────────────
    def _enter_window_mode(self):
        """Enter window control mode"""
        self.window_mode = True
        self._reset()
        print("[GESTURE] Entered Window Control Mode")

    # ── Gesture smoothing — majority vote ─────────────────────────────
    def _smooth(self):
        if not self._buf:
            return "No hand"
        counts = {}
        for g in self._buf:
            counts[g] = counts.get(g, 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] >= len(self._buf) // 2 + 1:
            return best
        return "Unknown"

    # ── IDLE / HOLDING state ──────────────────────────────────────────
    def _do_detection(self, frame, detected):
        # Special: Thumb Up in idle mode enters window mode
        if detected == "Thumb Up":
            self._enter_window_mode()
            return
        
        # Only proceed if gesture is in config and not handled above
        if detected not in self._valid_gestures:
            self._pending = None
            self._hold_start = 0.0
            return

        if self._pending != detected:
            self._pending    = detected
            self._hold_start = time.time()

        elapsed   = time.time() - self._hold_start
        remaining = max(0.0, config.GESTURE_HOLD_TIME - elapsed)
        dev, act  = config.GESTURE_COMMANDS[self._pending]

        cv2.putText(frame, f"Gesture: {self._pending}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.putText(frame, f"Hold {remaining:.1f}s...",
                    (20, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,100), 2)

        bw = frame.shape[1] - 40
        cv2.rectangle(frame, (20,208), (20+bw, 224), (60,60,60), -1)
        cv2.rectangle(frame, (20,208),
            (20 + int(bw * min(elapsed/config.GESTURE_HOLD_TIME, 1.0)), 224),
            (0,200,255), -1)

        cv2.putText(frame, f"-> {dev.upper()} {act.upper()}",
                    (20, 244), cv2.FONT_HERSH
