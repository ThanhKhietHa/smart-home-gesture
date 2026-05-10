"""
config.py — Central configuration for Smart Home Gesture Control
================================================================
Change settings HERE only. Do not edit other files for configuration.

Gesture flow (2-level menu):
  Level 1 — Entry gesture selects device:
    Open Palm   → Lights menu
    Peace Sign  → Door menu
    Pointing Up → AC menu
    Thumb Up    → Window menu   (long-hold GESTURE_HOLD_TIME)

  Level 2 — Action gesture inside menu:
    Thumb Up    → ON  / Roll Up / Confirm toggle
    Thumb Down  → OFF / Roll Down
    Open Palm   → Cancel (always exits back to idle)
"""

import os

# =====================================================================
# PATHS
# =====================================================================
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR        = os.path.join(BASE_DIR, 'models')
DATA_DIR          = os.path.join(BASE_DIR, 'data')

FACE_MODEL_PATH   = os.path.join(MODELS_DIR, 'face_landmarker.task')
HAND_MODEL_PATH   = os.path.join(MODELS_DIR, 'hand_landmarker.task')
ENROLLED_FILE     = os.path.join(DATA_DIR,   'enrolled_faces.pkl')
ENROLL_PHOTOS_DIR = os.path.join(DATA_DIR,   'enroll_photos')

os.makedirs(DATA_DIR,          exist_ok=True)
os.makedirs(ENROLL_PHOTOS_DIR, exist_ok=True)

# =====================================================================
# CAMERA — OPTIMIZED for Jetson Orin Nano
# =====================================================================
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 360
CAMERA_FPS    = 30

# =====================================================================
# PERFORMANCE OPTIMIZATIONS
# =====================================================================
# How often face recognition runs when LOCKED (every N frames)
FACE_PROCESS_EVERY_N_FRAMES_LOCKED    = 3
# How often FULL face recognition runs when UNLOCKED (presence check runs every frame)
FACE_PROCESS_EVERY_N_FRAMES_UNLOCKED  = 90
# MediaPipe model inference mutex — face+hand never run simultaneously
# (enforced in main.py via threading.Lock)

FACE_DETECTION_CONFIDENCE = 0.35
FACE_PRESENCE_CONFIDENCE  = 0.35

# =====================================================================
# MQTT — Shiftr.io cloud broker
# =====================================================================
MQTT_BROKER          = "khiet1111.cloud.shiftr.io"
MQTT_PORT            = 1883
MQTT_TOPIC_BASE      = "/smart_home/"
MQTT_USER            = "khiet1111"
MQTT_PASSWORD        = "khiet"
MQTT_RECONNECT_DELAY = 3.0

# =====================================================================
# FACE RECOGNITION
# =====================================================================
FACE_SHAPE_THRESHOLD    = 0.10
FACE_IDENTITY_THRESHOLD = 0.008

FACE_CONFIRM_FRAMES  = 5
FACE_RELOCK_FRAMES   = 25
FACE_ENROLL_TARGET   = 40
FACE_AUTH_TIMEOUT    = 300.0
FACE_MIN_HEIGHT_FRAC = 0.20

# =====================================================================
# GESTURE RECOGNITION
# =====================================================================
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE  = 0.4

# How long to hold entry gesture before entering device menu (seconds)
GESTURE_HOLD_TIME   = 1.5
# How long to hold action gesture (Thumb Up/Down) to confirm action
ACTION_HOLD_TIME    = 0.8
# Brief delay after entering menu before action gestures are accepted
#   (prevents accidental immediate confirm from entry gesture motion)
MENU_ENTRY_DELAY    = 0.5
# How long menu stays open with no valid gesture before auto-cancel
MENU_TIMEOUT        = 8.0

# =====================================================================
# DEVICE MENU DEFINITIONS
# 2-level gesture flow:
#   ENTRY_GESTURES: gesture → device label shown in menu
#   DEVICE_MENUS:   device  → { action_gesture: (mqtt_device, mqtt_action, display_label) }
#
# Open Palm is the universal cancel — handled in code, not listed here.
# =====================================================================

# Level 1 — which gesture opens which device menu
ENTRY_GESTURES = {
    "Open Palm":   "lights",
    "Peace Sign":  "door",
    "Pointing Up": "ac",
    "Thumb Up":    "window",   # long-hold → window menu
}

# Level 2 — inside each device menu
DEVICE_MENUS = {
    "lights": {
        "Thumb Up":   ("lights", "on",  "Lights ON"),
        "Thumb Down": ("lights", "off", "Lights OFF"),
    },
    "door": {
        "Thumb Up":   ("door", "toggle", "Door TOGGLE"),
    },
    "ac": {
        "Thumb Up":   ("ac", "on",  "AC ON"),
        "Thumb Down": ("ac", "off", "AC OFF"),
    },
    "window": {
        "Thumb Up":   ("window", "roll_up",   "Window UP"),
        "Thumb Down": ("window", "roll_down", "Window DOWN"),
    },
}

# Menu display names and entry hint text
DEVICE_DISPLAY = {
    "lights": "LIGHTS",
    "door":   "DOOR",
    "ac":     "AC",
    "window": "WINDOW",
}

DEVICE_ACTION_HINTS = {
    "lights": "Thumb UP = ON   |   Thumb DOWN = OFF   |   Open Palm = Cancel",
    "door":   "Thumb UP = Toggle   |   Open Palm = Cancel",
    "ac":     "Thumb UP = ON   |   Thumb DOWN = OFF   |   Open Palm = Cancel",
    "window": "Thumb UP = Roll Up  |  Thumb DOWN = Roll Down  |  Open Palm = Cancel",
}

# =====================================================================
# DEVICE INITIAL STATES (for on-screen panel)
# =====================================================================
DEVICE_INITIAL_STATES = {
    "lights": 0,
    "door":   0,
    "ac":     0,
    "window": "stopped",
}
