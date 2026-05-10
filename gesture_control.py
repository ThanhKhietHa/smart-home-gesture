"""
gesture_control.py — Hand Gesture Recognition (LIVE_STREAM mode)
================================================================
LIVE_STREAM:
  detect_async() costs only 0.35ms (non-blocking).
  Hand model result arrives via callback asynchronously.
  Most frames reuse the tracker's fast crop model (~10% of full cost)
  instead of running the full palm detector every frame.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import threading
import numpy as np
from collections import deque
import config

# =====================================================================
# LIVE_STREAM RESULT STORAGE
# =====================================================================
_hand_lock          = threading.Lock()
_latest_hand_result = None

def _hand_callback(result, image, timestamp_ms):
    global _latest_hand_result
    with _hand_lock:
        _latest_hand_result = result

# =====================================================================
# MEDIAPIPE — LIVE_STREAM mode
# =====================================================================
_hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=config.HAND_MODEL_PATH
    ),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=_hand_callback,
    num_hands=1,
    min_hand_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
)
_landmarker = vision.HandLandmarker.create_from_options(_hand_options)

_GS_IDLE    = "IDLE"
_GS_CONFIRM = "CONFIRM"

# =====================================================================
# GEOMETRY
# =====================================================================
def _dist(p1, p2):
    return math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)

# =====================================================================
# GESTURE DETECTION
# =====================================================================
def detect_gesture(lm):
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
        n  = sum([ie,me,re,pe])
        tp = _dist(thumb_tip, wrist)
        tv = thumb_tip.y - thumb_cmc.y

        if n == 0 and tp > 0.18:
            if tv < -0.10: return "Thumb Up"
            if tv >  0.10: return "Thumb Down"
        if n >= 3 and tp > 0.25: return "Spread"
        if n >= 3:               return "Open Palm"
        if n == 0 and tp < 0.22: return "Fist"
        if ie and me and re and not pe:     return "Three Fingers"
        if ie and me and not re and not pe: return "Peace Sign"
        if n == 4 and tp < 0.20:           return "Four Fingers"
        if _dist(thumb_tip, index_tip) < 0.06 and not me and not re:
            return "Pinch"
        if ie and not me and not re and not pe:
            il = _dist(index_tip, index_mcp)
            ml = _dist(middle_tip, middle_mcp)
            rl = _dist(ring_tip,   ring_mcp)
            if il > ml+0.03 and il > rl+0.03:
                v = index_tip.y - index_mcp.y
                if v >  -0.05: return "Pointing Down"
                if v <  -0.12: return "Pointing Up"
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
        self._ts           = 1000   # LIVE_STREAM timestamp, starts > 0

    # ── Main entry ────────────────────────────────────────────────────
    def process_frame(self, frame, mqtt, face_unlocked):
        """
        Sends frame to hand landmarker via detect_async (non-blocking).
        Reads latest result from callback.
        Returns (annotated_frame, feedback_or_None).
        """
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._ts += 33
            _landmarker.detect_async(mp_img, self._ts)
        except Exception:
            pass

        # Read latest result (may be 1 frame behind — fine for gestures)
        with _hand_lock:
            result = _latest_hand_result

        lm = result.hand_landmarks[0] if (result and result.hand_landmarks) else None

        self._buf.append(detect_gesture(lm))
        detected = self._smooth()

        # Draw landmarks
        if lm is not None:
            try:
                H, W = frame.shape[:2]
                for pt in lm:
                    cx = int(max(0, min(pt.x*W, W-1)))
                    cy = int(max(0, min(pt.y*H, H-1)))
                    cv2.circle(frame,(cx,cy),3,(0,255,0),-1)
            except Exception:
                pass

        # Blocked
        if not face_unlocked:
            self._reset()
            cv2.putText(frame,"FACE AUTH REQUIRED",
                        (20,160),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,220),2)
            self._draw_devices(frame)
            return frame, None

        feedback = None
        if self._state == _GS_CONFIRM:
            feedback = self._do_confirm(frame, detected, mqtt)
        else:
            self._do_detection(frame, detected)

        self._draw_devices(frame)
        return frame, feedback

    # ── Smoothing ─────────────────────────────────────────────────────
    def _smooth(self):
        if not self._buf: return "No hand"
        counts = {}
        for g in self._buf:
            counts[g] = counts.get(g,0)+1
        best = max(counts, key=counts.get)
        if counts[best] >= len(self._buf)//2+1:
            return best
        return "Unknown"

    # ── Detection / hold ──────────────────────────────────────────────
    def _do_detection(self, frame, detected):
        actionable = detected in config.GESTURE_COMMANDS
        if actionable:
            if self._pending != detected:
                self._pending    = detected
                self._hold_start = time.time()
            elapsed   = time.time()-self._hold_start
            remaining = max(0.0, config.GESTURE_HOLD_TIME-elapsed)
            dev, act  = config.GESTURE_COMMANDS[self._pending]
            cv2.putText(frame,f"Gesture: {self._pending}",
                        (20,160),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
            cv2.putText(frame,f"Hold {remaining:.1f}s...",
                        (20,196),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,100),2)
            bw = frame.shape[1]-40
            cv2.rectangle(frame,(20,208),(20+bw,224),(60,60,60),-1)
            cv2.rectangle(frame,(20,208),
                (20+int(bw*min(elapsed/config.GESTURE_HOLD_TIME,1.0)),224),
                (0,200,255),-1)
            cv2.putText(frame,f"-> {dev.upper()} {act.upper()}",
                        (20,244),cv2.FONT_HERSHEY_SIMPLEX,0.8,(180,255,180),2)
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
            if detected == "No hand":
                cv2.putText(frame,"No hand detected",
                            (20,160),cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,(80,80,80),1)
            else:
                cv2.putText(frame,f"Gesture: {detected}",
                            (20,160),cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,(120,120,120),1)

    # ── Confirm ───────────────────────────────────────────────────────
    def _do_confirm(self, frame, detected, mqtt):
        dev, act = config.GESTURE_COMMANDS.get(self._cur_gesture,("?","?"))
        cv2.rectangle(frame,(0,0),(frame.shape[1]-1,frame.shape[0]-1),(0,200,255),5)
        cv2.putText(frame,"CONFIRM ACTION?",
                    (20,155),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,255),2)
        cv2.putText(frame,f"{self._cur_gesture}  ->  {dev} {act}",
                    (20,192),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,0),2)
        cv2.putText(frame,"Thumb UP = YES     Thumb DOWN = NO",
                    (20,226),cv2.FONT_HERSHEY_SIMPLEX,0.75,(200,200,200),1)
        since = time.time()-self._conf_entry
        if since < config.CONFIRM_ENTRY_DELAY:
            rem = config.CONFIRM_ENTRY_DELAY-since
            cv2.putText(frame,f"Stabilising... {rem:.1f}s",
                        (20,260),cv2.FONT_HERSHEY_SIMPLEX,0.8,(160,160,160),1)
            self._conf_gesture = None; self._conf_start = 0.0
            return None
        if detected in ("Thumb Up","Thumb Down"):
            if self._conf_gesture != detected:
                self._conf_gesture = detected
                self._conf_start   = time.time()
            held = time.time()-self._conf_start
            rem  = max(0.0, config.CONFIRM_HOLD_TIME-held)
            bc   = (0,220,0) if detected=="Thumb Up" else (0,0,220)
            lbl  = "Thumb UP  (YES)" if detected=="Thumb Up" else "Thumb DOWN  (NO)"
            cv2.putText(frame,f"{lbl}   hold {rem:.1f}s",
                        (20,260),cv2.FONT_HERSHEY_SIMPLEX,0.82,bc,2)
            cv2.rectangle(frame,(20,272),(340,288),(60,60,60),-1)
            cv2.rectangle(frame,(20,272),
                (20+int(320*min(held/config.CONFIRM_HOLD_TIME,1.0)),288),bc,-1)
            if held >= config.CONFIRM_HOLD_TIME:
                if detected == "Thumb Up":
                    mqtt.publish(dev, act)
                    self._update_device(dev, act)
                    feedback = (f"{dev.upper()} {act.upper()} ACTIVATED!",(0,255,0))
                else:
                    feedback = ("Action CANCELLED",(0,0,255))
                self._reset()
                return feedback
        else:
            self._conf_gesture = None; self._conf_start = 0.0
            cv2.putText(frame,"Show Thumb UP or DOWN",
                        (20,260),cv2.FONT_HERSHEY_SIMPLEX,0.8,(180,180,180),1)
        return None

    def _update_device(self, device, action):
        if action == "toggle":
            cur = self.device_states.get(device,0)
            self.device_states[device] = 0 if cur else 1
        elif isinstance(self.device_states.get(device),str):
            self.device_states[device] = action
        else:
            self.device_states[device] = 1 if action=="on" else 0

    # ── Device panel (bottom-right, compact) ─────────────────────────
    def _draw_devices(self, frame):
        H, W = frame.shape[:2]
        n    = len(self.device_states)
        lh   = 20
        pw   = 160; ph = n*lh+8
        x0   = W-pw-4; y0 = H-ph-4
        overlay = frame.copy()
        cv2.rectangle(overlay,(x0,y0),(W-2,H-2),(15,15,15),-1)
        cv2.addWeighted(overlay,0.50,frame,0.50,0,frame)
        y = y0+lh
        for dev, state in self.device_states.items():
            if isinstance(state,str):
                label = state.upper()
                color = (0,200,0) if state not in ("stopped","off","close") \
                        else (80,80,80)
            else:
                label = "ON" if state else "OFF"
                color = (0,210,0) if state else (80,80,80)
            cv2.putText(frame,f"{dev.capitalize()}: {label}",
                        (x0+5,y),cv2.FONT_HERSHEY_SIMPLEX,0.48,color,1)
            y += lh

    def _reset(self):
        self._state        = _GS_IDLE
        self._pending      = None
        self._hold_start   = 0.0
        self._cur_gesture  = None
        self._conf_gesture = None
        self._conf_start   = 0.0
        self._buf.clear()

    def draw_fps(self, frame, fps):
        cv2.putText(frame,f"FPS: {fps:.1f}",
                    (frame.shape[1]-115,frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)
