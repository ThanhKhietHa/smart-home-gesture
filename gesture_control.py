"""
gesture_control.py — Optimized for Jetson Orin Nano
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
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

_GS_IDLE = "IDLE"
_GS_CONFIRM = "CONFIRM"

def _dist(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def detect_gesture(lm):
    try:
        if not lm or len(lm) < 21:
            return "No hand"

        wrist = lm[0]
        thumb_tip = lm[4]
        thumb_cmc = lm[1]
        index_tip = lm[8]
        index_mcp = lm[5]
        middle_tip = lm[12]
        middle_mcp = lm[9]
        ring_tip = lm[16]
        ring_mcp = lm[13]
        pinky_tip = lm[20]
        pinky_mcp = lm[17]

        def ext(tip, mcp, thr=0.06):
            return tip.y < mcp.y - thr

        ie = ext(index_tip, index_mcp)
        me = ext(middle_tip, middle_mcp)
        re = ext(ring_tip, ring_mcp)
        pe = ext(pinky_tip, pinky_mcp)
        n = sum([ie, me, re, pe])

        tp = _dist(thumb_tip, wrist)
        tv = thumb_tip.y - thumb_cmc.y

        if n == 0 and tp > 0.18:
            if tv < -0.10: return "Thumb Up"
            if tv > 0.10: return "Thumb Down"

        if n >= 3 and tp > 0.25: return "Spread"
        if n >= 3: return "Open Palm"
        if n == 0 and tp < 0.22: return "Fist"

        if ie and me and re and not pe: return "Three Fingers"
        if ie and me and not re and not pe: return "Peace Sign"
        if n == 4 and tp < 0.20: return "Four Fingers"
        if _dist(thumb_tip, index_tip) < 0.06 and not me and not re:
            return "Pinch"

        if ie and not me and not re and not pe:
            return "Pointing"

        return "Unknown"
    except:
        return "No hand"

class GestureControl:
    def __init__(self):
        self._state = _GS_IDLE
        self._pending = None
        self._hold_start = 0.0
        self._cur_gesture = None
        self._conf_gesture = None
        self._conf_start = 0.0
        self._conf_entry = 0.0
        self._buf = deque(maxlen=4)  # Smaller buffer = faster
        self.device_states = dict(config.DEVICE_INITIAL_STATES)

    def process_frame(self, frame, mqtt, face_unlocked):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = _landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            lm = result.hand_landmarks[0] if result.hand_landmarks else None
        except:
            lm = None

        self._buf.append(detect_gesture(lm))
        detected = self._smooth()

        if lm is not None:
            try:
                H, W = frame.shape[:2]
                for pt in lm:
                    cx = int(max(0, min(pt.x * W, W-1)))
                    cy = int(max(0, min(pt.y * H, H-1)))
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
            except:
                pass

        if not face_unlocked:
            self._reset()
            cv2.putText(frame, "FACE LOCKED", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)
            self._draw_devices(frame)
            return frame, None

        feedback = None
        if self._state == _GS_CONFIRM:
            feedback = self._do_confirm(frame, detected, mqtt)
        else:
            self._do_detection(frame, detected)

        self._draw_devices(frame)
        return frame, feedback

    def _smooth(self):
        if not self._buf:
            return "No hand"
        counts = {}
        for g in self._buf:
            counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get)

    def _do_detection(self, frame, detected):
        actionable = detected in config.GESTURE_COMMANDS

        if actionable:
            if self._pending != detected:
                self._pending = detected
                self._hold_start = time.time()

            elapsed = time.time() - self._hold_start
            remaining = max(0.0, config.GESTURE_HOLD_TIME - elapsed)
            dev, act = config.GESTURE_COMMANDS[self._pending]

            cv2.putText(frame, f"{self._pending}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            cv2.putText(frame, f"Hold {remaining:.1f}s", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,100), 2)

            if elapsed >= config.GESTURE_HOLD_TIME:
                self._cur_gesture = self._pending
                self._state = _GS_CONFIRM
                self._conf_entry = time.time()
                self._pending = None
        else:
            self._pending = None

    def _do_confirm(self, frame, detected, mqtt):
        dev, act = config.GESTURE_COMMANDS.get(self._cur_gesture, ("?", "?"))

        cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0,200,255), 4)
        cv2.putText(frame, f"{self._cur_gesture} -> {dev} {act}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(frame, "Thumb UP=YES  DOWN=NO", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        if time.time() - self._conf_entry < config.CONFIRM_ENTRY_DELAY:
            return None

        if detected in ("Thumb Up", "Thumb Down"):
            if self._conf_gesture != detected:
                self._conf_gesture = detected
                self._conf_start = time.time()

            held = time.time() - self._conf_start
            if held >= config.CONFIRM_HOLD_TIME:
                if detected == "Thumb Up":
                    mqtt.publish(dev, act)
                    self._update_device(dev, act)
                    feedback = (f"{dev.upper()} {act.upper()}!", (0,255,0))
                else:
                    feedback = ("CANCELLED", (0,0,255))
                self._reset()
                return feedback
            else:
                rem = config.CONFIRM_HOLD_TIME - held
                cv2.putText(frame, f"Hold {rem:.1f}s", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        else:
            self._conf_gesture = None

        return None

    def _update_device(self, device, action):
        if action == "toggle":
            cur = self.device_states.get(device, 0)
            self.device_states[device] = 0 if cur else 1
        elif isinstance(self.device_states.get(device), str):
            self.device_states[device] = action
        else:
            self.device_states[device] = 1 if action == "on" else 0

    def _draw_devices(self, frame):
        H, W = frame.shape[:2]
        n = len(self.device_states)
        x0 = W - 130
        y0 = H - (n * 16 + 10)
        y = y0 + 12
        for dev, state in self.device_states.items():
            if isinstance(state, str):
                label = state.upper()
                color = (0,200,0) if state not in ("stopped","off") else (80,80,80)
            else:
                label = "ON" if state else "OFF"
                color = (0,210,0) if state else (80,80,80)
            cv2.putText(frame, f"{dev[:4]}:{label}", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 14

    def _reset(self):
        self._state = _GS_IDLE
        self._pending = None
        self._cur_gesture = None
        self._conf_gesture = None
        self._buf.clear()

    def draw_fps(self, frame, fps):
        cv2.putText(frame, f"{fps:.1f}fps", (frame.shape[1]-65, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
