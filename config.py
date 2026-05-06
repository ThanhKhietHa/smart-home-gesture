"""
config.py — Central configuration for Smart Home Gesture Control
Optimized for Jetson Orin Nano
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
# CAMERA
# =====================================================================
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640   # Keep at 320x240 for balance
CAMERA_HEIGHT = 360   # Change to 160x120 for 25+ FPS
CAMERA_FPS    = 30

# =====================================================================
# PERFORMANCE OPTIMIZATIONS (NEW)
# =====================================================================
# Frame skipping - Run detection less often for higher FPS
FACE_PROCESS_EVERY_N_FRAMES_LOCKED = 2      # Process every 2nd frame when locked
FACE_PROCESS_EVERY_N_FRAMES_UNLOCKED = 90   # Process every 90th frame when unlocked
GESTURE_PROCESS_EVERY_N_FRAMES = 2          # Process every 2nd frame when unlocked

# MediaPipe confidence (lower = faster, slightly less accurate)
FACE_DETECTION_CONFIDENCE = 0.35   # Was 0.4
FACE_PRESENCE_CONFIDENCE = 0.35    # Was 0.4
HAND_DETECTION_CONFIDENCE = 0.5    # Keep as is
HAND_TRACKING_CONFIDENCE = 0.4     # Was 0.5

# =====================================================================
# MQTT
# =====================================================================
MQTT_BROKER         = "localhost"
MQTT_PORT           = 1883
MQTT_TOPIC_BASE     = "/smart_home/"
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
GESTURE_HOLD_TIME    = 2.0
CONFIRM_HOLD_TIME    = 0.8
CONFIRM_ENTRY_DELAY  = 0.8

# =====================================================================
# DEVICE → GESTURE MAPPING
# =====================================================================
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

DEVICE_INITIAL_STATES = {
    "lights":   0,
    "fan":      0,
    "door":     0,
    "ac":       0,
    "curtains": "stopped",
    "window":   "stopped",
}
