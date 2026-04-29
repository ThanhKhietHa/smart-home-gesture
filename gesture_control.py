"""
gesture_control.py — Hand Gesture Recognition & Device Control
==============================================================
Handles:
  - All static gestures: Open Palm, Fist, Peace Sign, Thumb Up/Down, etc.
  - Hold-to-confirm system before sending ANY command (including window)
  - Window: Thumb Up/Down held 2s → confirm screen → Thumb Up 0.8s = sent
  - Device state tracking for UI display
  - BLOCKED when face not authenticated

Flow:  IDLE → HOLDING (2s bar) → CONFIRM → executes/cancels → IDLE
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math, time
import numpy as np
from collections import deque
import config

# ── MediaPipe ─────────────────────────────────────────────────────────
_hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=config.HAND_MODEL_PATH,
        delegate=python.BaseOptions.Delegate.CPU  # Added comma here
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
)
_landmarker = vision.HandLandmarker.create_from_options(_hand_options)

_GS_IDLE    = "IDLE"
_GS_CONFIRM = "CONFIRM"

# ── Geometry helper ───────────────────────────────────────────────────
def _dist(p1, p2):
    return math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)

# ── Gesture detection ─────────────────────────────────────────────────
def detect_gesture(lm):
    if not lm:
        return "No hand"

    wrist      = lm[0];  thumb_tip = lm[4];  thumb_cmc = lm[1]
    index_tip  = lm[8];  middle_tip= lm[12]; ring_tip  = lm[16]; pinky_tip = lm[20]
    index_mcp  = lm[5];  middle_mcp= lm[9];  ring_mcp  = lm[13]; pinky_mcp = lm[17]

    def ext(tip, mcp, thr=0.07):
        return tip.y < mcp.y - thr

    ie = ext(index_tip, index_mcp);  me = ext(middle_tip, middle_mcp)
    re = ext(ring_tip,  ring_mcp);   pe = ext(pinky_tip,  pinky_mcp)
    n  = sum([ie, me, re, pe])

    tp = _dist(thumb_tip, wrist)          # thumb-to-palm distance
    tv = thumb_tip.y - thumb_cmc.y        # negative = pointing up

    # Thumb gestures (all fingers folded)
    if n == 0 and tp > 0.18:
        if tv < -0.10: return "Thumb Up"
        if tv >  0.10: return "Thumb Down"

    if n >= 3 and tp > 0.25: return "Spread"
    if n >= 3:               return "Open Palm"
    if n == 0 and tp < 0.22: return "Fist"
    if ie and me and re and not pe: return "Three Fingers"
    if ie and me and not re and not pe: return "Peace Sign"
    if n == 4 and tp < 0.20: return "Four Fingers"
    if _dist(thumb_tip, index_tip) < 0.06 and not me and not re: return "Pinch"

    # Pointing (index only)
    if ie and not me and not re and not pe:
        il = _dist(index_tip, index_mcp)
        if il > _dist(middle_tip,middle_mcp)+0.03 and il > _dist(ring_tip,ring_mcp)+0.03:
            v = index_tip.y - index_mcp.y
            if v >  -0.05: return "Pointing Down"
            if v <  -0.12: return "Pointing Up"

    return "Unknown"

# ── Controller ────────────────────────────────────────────────────────
class GestureControl:
    def __init__(self):
        self._state           = _GS_IDLE
        self._pending         = None
        self._hold_start      = 0.0
        self._cur_gesture     = None
        self._conf_gesture    = None
        self._conf_start      = 0.0
        self._conf_entry      = 0.0
        self._buf             = deque(maxlen=4)
        self.device_states    = dict(config.DEVICE_INITIAL_STATES)

    def process_frame(self, frame, mqtt, face_unlocked):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        lm     = result.hand_landmarks[0] if result.hand_landmarks else None
        self._buf.append(detect_gesture(lm))
        detected = self._smooth()

        # Draw landmarks
        if lm:
            H, W = frame.shape[:2]
            for pt in lm:
                cv2.circle(frame,(int(pt.x*W),int(pt.y*H)),5,(0,255,0),-1)

        # Blocked
        if not face_unlocked:
            self._reset()
            cv2.putText(frame,"FACE AUTH REQUIRED",
                        (20,160),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,220),2)
            self._draw_devices(frame)
            return frame

        if self._state == _GS_CONFIRM:
            self._do_confirm(frame, detected, mqtt)
        else:
            self._do_detection(frame, detected)

        self._draw_devices(frame)
        return frame

    def _smooth(self):
        if not self._buf: return "No hand"
        c = {}
        for g in self._buf: c[g] = c.get(g,0)+1
        best = max(c, key=c.get)
        return best if c[best] >= len(self._buf)//2+1 else "Unknown"

    def _do_detection(self, frame, detected):
        actionable = detected in config.GESTURE_COMMANDS

        if actionable:
            if self._pending != detected:
                self._pending    = detected
                self._hold_start = time.time()

            elapsed   = time.time() - self._hold_start
            remaining = max(0.0, config.GESTURE_HOLD_TIME - elapsed)
            dev, act  = config.GESTURE_COMMANDS[self._pending]

            cv2.putText(frame, f"Gesture: {self._pending}",
                        (20,160),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,255),2)
            cv2.putText(frame, f"Hold {remaining:.1f}s to confirm...",
                        (20,198),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,100),2)
            bw = frame.shape[1]-40
            cv2.rectangle(frame,(20,210),(20+bw,228),(60,60,60),-1)
            cv2.rectangle(frame,(20,210),
                (20+int(bw*min(elapsed/config.GESTURE_HOLD_TIME,1.0)),228),(0,200,255),-1)
            cv2.putText(frame, f"->  {dev.upper()}  {act.upper()}",
                        (20,248),cv2.FONT_HERSHEY_SIMPLEX,0.8,(180,255,180),2)

            if elapsed >= config.GESTURE_HOLD_TIME:
                self._cur_gesture = self._pending
                self._state       = _GS_CONFIRM
                self._conf_entry  = time.time()
                self._conf_gesture= None
                self._conf_start  = 0.0
                self._pending     = None
                self._hold_start  = 0.0
        else:
            self._pending    = None
            self._hold_start = 0.0
            msg = detected if detected=="No hand" else f"Gesture: {detected}"
            cv2.putText(frame, msg,(20,160),cv2.FONT_HERSHEY_SIMPLEX,0.95,(120,120,120),2)

    def _do_confirm(self, frame, detected, mqtt):
        dev, act = config.GESTURE_COMMANDS.get(self._cur_gesture, ("?","?"))

        cv2.rectangle(frame,(0,0),(frame.shape[1],frame.shape[0]),(0,200,255),5)
        cv2.putText(frame,"CONFIRM ACTION?",
                    (20,155),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
        cv2.putText(frame,f"  {self._cur_gesture}  ->  {dev} {act}",
                    (20,193),cv2.FONT_HERSHEY_SIMPLEX,0.95,(255,255,0),2)
        cv2.putText(frame,"Thumb UP = YES     Thumb DOWN = NO",
                    (20,228),cv2.FONT_HERSHEY_SIMPLEX,0.78,(200,200,200),2)

        since = time.time()-self._conf_entry
        if since < config.CONFIRM_ENTRY_DELAY:
            cv2.putText(frame,f"Stabilising... {config.CONFIRM_ENTRY_DELAY-since:.1f}s",
                        (20,262),cv2.FONT_HERSHEY_SIMPLEX,0.8,(160,160,160),2)
            self._conf_gesture = None;  self._conf_start = 0.0
            return

        if detected in ("Thumb Up","Thumb Down"):
            if self._conf_gesture != detected:
                self._conf_gesture = detected;  self._conf_start = time.time()
            held = time.time()-self._conf_start
            rem  = max(0.0, config.CONFIRM_HOLD_TIME-held)
            bc   = (0,220,0) if detected=="Thumb Up" else (0,0,220)
            lbl  = "Thumb UP  (YES)" if detected=="Thumb Up" else "Thumb DOWN  (NO)"

            cv2.putText(frame,f"{lbl}   hold {rem:.1f}s",
                        (20,262),cv2.FONT_HERSHEY_SIMPLEX,0.85,bc,2)
            cv2.rectangle(frame,(20,275),(340,293),(60,60,60),-1)
            cv2.rectangle(frame,(20,275),
                (20+int(320*min(held/config.CONFIRM_HOLD_TIME,1)),293),bc,-1)

            if held >= config.CONFIRM_HOLD_TIME:
                if detected=="Thumb Up":
                    mqtt.publish(dev, act)
                    self._update_device(dev, act)
                    msg, mc = f"{dev.upper()} {act.upper()} ACTIVATED!", (0,255,0)
                else:
                    msg, mc = "Action CANCELLED", (0,0,255)
                cv2.putText(frame, msg,(20,322),cv2.FONT_HERSHEY_SIMPLEX,1.0,mc,3)
                cv2.imshow('Smart Home', frame);  cv2.waitKey(1200)
                self._reset()
        else:
            self._conf_gesture = None;  self._conf_start = 0.0
            cv2.putText(frame,"Show Thumb UP or DOWN",
                        (20,262),cv2.FONT_HERSHEY_SIMPLEX,0.8,(180,180,180),2)

    def _update_device(self, device, action):
        if action == "toggle":
            self.device_states[device] = 0 if self.device_states.get(device,0) else 1
        elif isinstance(self.device_states.get(device), str):
            self.device_states[device] = action
        else:
            self.device_states[device] = 1 if action=="on" else 0

    def _draw_devices(self, frame):
        H = frame.shape[0]
        y = H - 10 - len(self.device_states)*28
        cv2.rectangle(frame,(0,y-10),(210,H),(20,20,20),-1)
        for dev, state in self.device_states.items():
            if isinstance(state, str):
                label = state.upper()
                color = (0,200,0) if state not in ("stopped","off","close") else (80,80,80)
            else:
                label = "ON" if state else "OFF"
                color = (0,220,0) if state else (80,80,80)
            cv2.putText(frame,f"{dev.capitalize()}: {label}",
                        (8,y),cv2.FONT_HERSHEY_SIMPLEX,0.62,color,1)
            y += 28

    def _reset(self):
        self._state=_GS_IDLE; self._pending=None; self._hold_start=0.0
        self._cur_gesture=None; self._conf_gesture=None
        self._conf_start=0.0;  self._buf.clear()

    def draw_fps(self, frame, fps):
        cv2.putText(frame,f"FPS: {fps:.1f}",
                    (frame.shape[1]-110,frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)
