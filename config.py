"""
config.py — Optimized for Jetson Orin Nano (JetPack 6.0)
Target: 18-22 FPS
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
# CAMERA - CRITICAL FOR FPS
# =====================================================================
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 320    # 320x240 - OPTIMAL (DO NOT CHANGE to 640x480)
CAMERA_HEIGHT = 240    # This gives 2-3x speedup
CAMERA_FPS    = 30

# Frame skipping - KEY OPTIMIZATION
FACE_DETECT_EVERY_N_FRAMES = 2          # Run face detect every 2 frames when locked
FACE_DETECT_EVERY_N_FRAMES_UNLOCKED = 90 # Run every 90 frames when unlocked
GESTURE_EVERY_N_FRAMES = 2               # Run gesture every 2 frames when unlocked

# =====================================================================
# MQTT
# =====================================================================
MQTT_BROKER         = "localhost"
MQTT_PORT           = 1883
MQTT_TOPIC_BASE     = "/smart_home/"
MQTT_RECONNECT_DELAY = 3.0

# =====================================================================
# FACE RECOGNITION - RELAXED THRESHOLDS
# =====================================================================
FACE_SHAPE_THRESHOLD    = 0.10
FACE_IDENTITY_THRESHOLD = 0.008

FACE_CONFIRM_FRAMES  = 4       # Faster unlock
FACE_RELOCK_FRAMES   = 25
FACE_ENROLL_TARGET   = 25      # Reduced for faster enrollment
FACE_AUTH_TIMEOUT    = 300.0
FACE_MIN_HEIGHT_FRAC = 0.18    # Relaxed requirement

# MediaPipe confidence (lower = faster)
FACE_DETECTION_CONFIDENCE = 0.35   # Default was 0.5
FACE_PRESENCE_CONFIDENCE = 0.35    # Default was 0.5

# =====================================================================
# GESTURE RECOGNITION - FASTER
# =====================================================================
HAND_DETECTION_CONFIDENCE = 0.5    # Reduced from 0.6
HAND_TRACKING_CONFIDENCE  = 0.4    # Reduced from 0.5

GESTURE_HOLD_TIME    = 1.2   # Faster response
CONFIRM_HOLD_TIME    = 0.5   # Faster confirmation
CONFIRM_ENTRY_DELAY  = 0.5   # Faster entry

# =====================================================================
# DEVICE MAPPING
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
