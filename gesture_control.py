"""
gesture_control.py — Hand Gesture Recognition & Device Control
==============================================================
2-Level Menu System:
  Level 1 (IDLE):    Hold entry gesture 1.5s → enter device menu
  Level 2 (IN_MENU): Show action options for that device
                     Thumb Up / Thumb Down = action (hold 0.8s)
                     Open Palm             = cancel (always)
  Timeout: menu auto-cancels after MENU_TIMEOUT seconds
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

# =====================================================================
# STATES
# =====================================================================
_GS_IDLE = "IDLE"
_GS_MENU = "MENU"

# =====================================================================
# HELPERS
# =====================================================================
def _dist(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )

# =====================================================================
# GESTURE DETECTION
# =====================================================================
def detect_gesture(lm):
    try:
        if not lm or len(lm) < 21:
            return "No hand"

        wrist      = lm[0]
        thumb_tip  = lm[4]
        thumb_cmc  = lm[1]

        index_tip  = lm[8]
        index_mcp  = lm[5]

        middle_tip = lm[12]
        middle_mcp = lm[9]

        ring_tip   = lm[16]
        ring_mcp   = lm[13]

        pinky_tip  = lm[20]
        pinky_mcp  = lm[17]

        def ext(tip, mcp, thr=0.07):
            return tip.y < mcp.y - thr

        ie = ext(index_tip, index_mcp)
        me = ext(middle_tip, middle_mcp)
        re = ext(ring_tip, ring_mcp)
        pe = ext(pinky_tip, pinky_mcp)

        n = sum([ie, me, re, pe])

        tp = _dist(thumb_tip, wrist)
        tv = thumb_tip.y - thumb_cmc.y

        # Thumb Up / Down
        if n == 0 and tp > 0.18:
            if tv < -0.10:
                return "Thumb Up"
            if tv > 0.10:
                return "Thumb Down"

        # Open Palm
        if n >= 3:
            return "Open Palm"

        # Fist
        if n == 0 and tp < 0.22:
            return "Fist"

        # Peace Sign
        if ie and me and not re and not pe:
            return "Peace Sign"

        # Pointing Up
        if ie and not me and not re and not pe:
            il = _dist(index_tip, index_mcp)
            ml = _dist(middle_tip, middle_mcp)
            rl = _dist(ring_tip, ring_mcp)

            if il > ml + 0.03 and il > rl + 0.03:
                v = index_tip.y - index_mcp.y
                if v < -0.12:
                    return "Pointing Up"

        return "Unknown"

    except Exception:
        return "No hand"

# =====================================================================
# GESTURE CONTROL
# =====================================================================
class GestureControl:

    def __init__(self):

        self._state = _GS_IDLE

        self._active_device = None

        self._hold_gesture = None
        self._hold_start = 0.0

        self._action_gesture = None
        self._action_start = 0.0

        self._menu_entry_time = 0.0

        self._buf = deque(maxlen=7)

        self.device_states = dict(config.DEVICE_INITIAL_STATES)

        self._entry_gestures = set(config.ENTRY_GESTURES.keys())

    # =================================================================
    # MAIN
    # =================================================================
    def process_frame(self, frame, mqtt, face_unlocked):

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = _landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
            )

            lm = result.hand_landmarks[0] if result.hand_landmarks else None

        except Exception:
            lm = None

        self._buf.append(detect_gesture(lm))
        detected = self._smooth()

        if lm is not None:
            self._draw_skeleton(frame, lm, detected)

        if not face_unlocked:
            self._reset()

            cv2.putText(
                frame,
                "FACE AUTH REQUIRED",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 220),
                2
            )

            self._draw_devices(frame)

            return frame, None

        feedback = None

        if self._state == _GS_IDLE:
            self._do_idle(frame, detected)

        elif self._state == _GS_MENU:
            feedback = self._do_menu(frame, detected, mqtt)

        self._draw_devices(frame)

        return frame, feedback

    # =================================================================
    # IDLE
    # =================================================================
    def _do_idle(self, frame, detected):

        if detected in self._entry_gestures:

            if self._hold_gesture != detected:
                self._hold_gesture = detected
                self._hold_start = time.time()

            elapsed = time.time() - self._hold_start

            remaining = max(
                0.0,
                config.GESTURE_HOLD_TIME - elapsed
            )

            device = config.ENTRY_GESTURES[detected]

            dev_label = config.DEVICE_DISPLAY.get(
                device,
                device.upper()
            )

            cv2.putText(
                frame,
                detected,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Hold to open {dev_label} menu... {remaining:.1f}s",
                (20, 194),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 220, 80),
                2
            )

            bw = frame.shape[1] - 40

            cv2.rectangle(
                frame,
                (20, 206),
                (20 + bw, 220),
                (60, 60, 60),
                -1
            )

            prog = min(
                elapsed / config.GESTURE_HOLD_TIME,
                1.0
            )

            cv2.rectangle(
                frame,
                (20, 206),
                (20 + int(bw * prog), 220),
                (0, 200, 255),
                -1
            )

            if elapsed >= config.GESTURE_HOLD_TIME:

                self._active_device = device
                self._state = _GS_MENU

                self._menu_entry_time = time.time()

                self._action_gesture = None
                self._action_start = 0.0

                self._hold_gesture = None
                self._hold_start = 0.0

        else:

            if self._hold_gesture is not None:
                self._hold_gesture = None
                self._hold_start = 0.0

    # =================================================================
    # MENU
    # =================================================================
    def _do_menu(self, frame, detected, mqtt):

        device = self._active_device

        dev_label = config.DEVICE_DISPLAY.get(
            device,
            device.upper()
        )

        hint = config.DEVICE_ACTION_HINTS.get(device, "")

        actions = config.DEVICE_MENUS.get(device, {})

        now = time.time()

        if now - self._menu_entry_time > config.MENU_TIMEOUT:
            self._reset()
            return None

        time_left = config.MENU_TIMEOUT - (
            now - self._menu_entry_time
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1]-1, frame.shape[0]-1),
            (0, 200, 255),
            5
        )

        cv2.putText(
            frame,
            f"{dev_label} MENU",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            hint,
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (200, 200, 200),
            1
        )

        cv2.putText(
            frame,
            f"Auto-cancel in {time_left:.0f}s",
            (20, 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (130, 130, 130),
            1
        )

        since_entry = now - self._menu_entry_time

        if since_entry < config.MENU_ENTRY_DELAY:

            rem = config.MENU_ENTRY_DELAY - since_entry

            cv2.putText(
                frame,
                f"Stabilising... {rem:.1f}s",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (160, 160, 160),
                1
            )

            self._action_gesture = None
            self._action_start = 0.0

            return None

        if detected in actions:

            if self._action_gesture != detected:
                self._action_gesture = detected
                self._action_start = now

            held = now - self._action_start

            remaining = max(
                0.0,
                config.ACTION_HOLD_TIME - held
            )

            _, _, lbl = actions[detected]

            bar_color = (
                (0, 220, 0)
                if detected == "Thumb Up"
                else (0, 80, 220)
            )

            cv2.putText(
                frame,
                f"{detected} → {lbl}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 80),
                2
            )

            cv2.putText(
                frame,
                f"Hold {remaining:.1f}s to confirm",
                (20, 234),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                bar_color,
                2
            )

            bw = frame.shape[1] - 40

            cv2.rectangle(
                frame,
                (20, 246),
                (20 + bw, 260),
                (60, 60, 60),
                -1
            )

            prog = min(
                held / config.ACTION_HOLD_TIME,
                1.0
            )

            cv2.rectangle(
                frame,
                (20, 246),
                (20 + int(bw * prog), 260),
                bar_color,
                -1
            )

            if held >= config.ACTION_HOLD_TIME:

                mqtt_device, mqtt_action, lbl = actions[detected]

                mqtt.publish(mqtt_device, mqtt_action)

                self._update_device(
                    mqtt_device,
                    mqtt_action
                )

                feedback = (
                    f"{lbl} ACTIVATED!",
                    (0, 255, 0)
                )

                self._reset()

                return feedback

        else:

            if self._action_gesture is not None:
                self._action_gesture = None
                self._action_start = 0.0

        return None

    # =================================================================
    # SMOOTHING
    # =================================================================
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

    # =================================================================
    # DRAW HAND
    # =================================================================
    def _draw_skeleton(self, frame, lm, detected):

        H, W = frame.shape[:2]

        in_menu = self._state == _GS_MENU

        if in_menu and detected in config.DEVICE_MENUS.get(
            self._active_device,
            {}
        ):
            color = (0, 255, 0)

        elif detected in self._entry_gestures:
            color = (0, 255, 255)

        else:
            color = (90, 90, 90)

        for pt in lm:

            cx = int(max(0, min(pt.x * W, W - 1)))
            cy = int(max(0, min(pt.y * H, H - 1)))

            cv2.circle(
                frame,
                (cx, cy),
                3,
                color,
                -1
            )

    # =================================================================
    # DEVICE UPDATE
    # =================================================================
    def _update_device(self, device, action):

        if action == "toggle":

            cur = self.device_states.get(device, 0)

            self.device_states[device] = 0 if cur else 1

        elif isinstance(self.device_states.get(device), str):

            self.device_states[device] = action

        else:

            self.device_states[device] = (
                1 if action in ("on", "roll_up")
                else 0
            )

    # =================================================================
    # DEVICE PANEL
    # =================================================================
    def _draw_devices(self, frame):

        H, W = frame.shape[:2]

        items = list(self.device_states.items())

        n = len(items)

        lh = 20
        pw = 155

        ph = n * lh + 8

        x0 = W - pw - 4
        y0 = H - ph - 4

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x0, y0),
            (W - 2, H - 2),
            (15, 15, 15),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.50,
            frame,
            0.50,
            0,
            frame
        )

        y = y0 + lh

        for dev, state in items:

            is_active = (
                self._state == _GS_MENU and
                dev == self._active_device
            )

            if isinstance(state, str):

                label = state.upper()

                color = (
                    (0, 200, 0)
                    if state not in (
                        "stopped",
                        "off",
                        "close",
                        "roll_down"
                    )
                    else (80, 80, 80)
                )

            else:

                label = "ON" if state else "OFF"

                color = (
                    (0, 210, 0)
                    if state
                    else (80, 80, 80)
                )

            if is_active:
                color = (0, 220, 255)

            cv2.putText(
                frame,
                f"{dev.capitalize()}: {label}",
                (x0 + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1
            )

            y += lh

    # =================================================================
    # RESET
    # =================================================================
    def reset(self):
        self._reset()

    def _reset(self):

        self._state = _GS_IDLE

        self._active_device = None

        self._hold_gesture = None
        self._hold_start = 0.0

        self._action_gesture = None
        self._action_start = 0.0

        self._menu_entry_time = 0.0

        self._buf.clear()

    # =================================================================
    # FPS
    # =================================================================
    def draw_fps(self, frame, fps):

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 115, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1
        )
