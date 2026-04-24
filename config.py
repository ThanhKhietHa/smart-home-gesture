"""
config.py — Central configuration for Smart Home Gesture Control
================================================================
Change settings HERE only. Do not edit other files for configuration.
"""

import os

# =====================================================================
# PATHS
# =====================================================================
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR          = os.path.join(BASE_DIR, 'models')
DATA_DIR            = os.path.join(BASE_DIR, 'data')

FACE_MODEL_PATH     = os.path.join(MODELS_DIR, 'face_landmarker.task')
HAND_MODEL_PATH     = os.path.join(MODELS_DIR, 'hand_landmarker.task')
ENROLLED_FILE       = os.path.join(DATA_DIR,   'enrolled_faces.pkl')
ENROLL_PHOTOS_DIR   = os.path.join(DATA_DIR,   'enroll_photos')

os.makedirs(DATA_DIR,          exist_ok=True)
os.makedirs(ENROLL_PHOTOS_DIR, exist_ok=True)

# =====================================================================
# CAMERA
# =====================================================================
CAMERA_INDEX        = 0       # 0 = default camera. Change to 1 if using 2nd camera.
CAMERA_WIDTH        = 640
CAMERA_HEIGHT       = 480

# =====================================================================
# MQTT
# =====================================================================
# If Mosquitto is running on the Jetson itself → keep "localhost"
# If running on another device → change to that device's IP e.g. "192.168.1.50"
MQTT_BROKER         = "localhost"
MQTT_PORT           = 1883
MQTT_TOPIC_BASE     = "/smart_home/"
MQTT_RECONNECT_DELAY = 3.0    # seconds between reconnect attempts

# =====================================================================
# FACE RECOGNITION
# =====================================================================
# --- Thresholds (close range 30-50cm, 2-3 family members) ---
# Lower = stricter.
# TUNING GUIDE:
#   1. Enroll yourself, watch Shape/Cosine values in debug panel
#   2. Note the highest values YOUR face produces (e.g. 0.18 / 0.08)
#   3. Enroll a family member, note their values when THEIR face scans
#   4. Set thresholds BETWEEN your max and their min
#   5. If getting false rejects  → raise by 0.02
#   6. If getting false accepts  → lower by 0.02
FACE_SHAPE_THRESHOLD    = 0.32
FACE_IDENTITY_THRESHOLD = 0.14

FACE_CONFIRM_FRAMES     = 6     # consecutive match frames needed to unlock
FACE_RELOCK_FRAMES      = 30    # consecutive no-match frames to re-lock
FACE_ENROLL_TARGET      = 40    # good frames collected per person during enroll
FACE_AUTH_TIMEOUT       = 300.0 # seconds — auto re-lock after this idle time
FACE_MIN_HEIGHT_FRAC    = 0.25  # face must fill at least 25% of frame height

# =====================================================================
# GESTURE RECOGNITION
# =====================================================================
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE  = 0.6

# Hold time before gesture enters confirm mode
GESTURE_HOLD_TIME       = 2.0   # seconds

# Confirm mode settings
CONFIRM_HOLD_TIME       = 0.8   # seconds to hold thumb up/down
CONFIRM_ENTRY_DELAY     = 0.8   # cooldown after entering confirm mode

# Window rolling — uses same hold-to-confirm system as other gestures
# Thumb Up held for GESTURE_HOLD_TIME → confirm screen → Thumb Up 0.8s = roll_up sent once
# Thumb Down held for GESTURE_HOLD_TIME → confirm screen → Thumb Up 0.8s = roll_down sent once
# This matches how all other device commands work.

# =====================================================================
# DEVICE → GESTURE MAPPING
# =====================================================================
# Format: "Gesture Name": ("device", "action")
# Devices: lights, fan, door, ac, curtains, window
# Actions: on, off, toggle, open, close, stop, roll_up, roll_down
#
# NOTE: Window (roll_up / roll_down) is handled separately in gesture_control.py
#       using the continuous hold system — do NOT add window here.
GESTURE_COMMANDS = {
    "Open Palm":     ("lights",   "on"),
    "Fist":          ("lights",   "off"),
    "Peace Sign":    ("door",     "toggle"),
    "Pointing Up":   ("ac",       "off"),
    "Pointing Down": ("ac",       "on"),
    "Three Fingers": ("fan",      "on"),
    "Four Fingers":  ("fan",      "off"),
    "Pinch":         ("curtains", "open"),
    "Spread":        ("curtains", "close"),
    "Thumb Up":      ("window",   "roll_up"),
    "Thumb Down":    ("window",   "roll_down"),
}

# Initial device states shown on UI
# 0 = OFF,  1 = ON,  string = custom state (for curtains/window)
DEVICE_INITIAL_STATES = {
    "lights":   0,
    "fan":      0,
    "door":     0,
    "ac":       0,
    "curtains": "stopped",
    "window":   "stopped",
}
