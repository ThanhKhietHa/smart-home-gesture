"""
config.py — Central configuration for Smart Home Gesture Control
================================================================
Change settings HERE only. Do not edit other files for configuration.
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
CAMERA_WIDTH  = 640    # Changed from 640x360 (640x360 is too slow)
CAMERA_HEIGHT = 360   # 320x240 gives 2x better FPS
CAMERA_FPS    = 30

# =====================================================================
# PERFORMANCE OPTIMIZATIONS (Add these for better FPS)
# =====================================================================
# Frame skipping - run detection less often for higher FPS
FACE_PROCESS_EVERY_N_FRAMES_LOCKED = 2      # Process every 2nd frame when locked
FACE_PROCESS_EVERY_N_FRAMES_UNLOCKED = 90   # Process every 90th frame when unlocked
GESTURE_PROCESS_EVERY_N_FRAMES = 2          # Process every 2nd frame when unlocked

# MediaPipe confidence (lower = faster, slightly less accurate)
FACE_DETECTION_CONFIDENCE = 0.35   # Default was 0.5
FACE_PRESENCE_CONFIDENCE = 0.35    # Default was 0.5

# =====================================================================
# MQTT — Shiftr.io cloud broker (FIXED)
# =====================================================================
# IMPORTANT: Your broker address MUST be: your-namespace.cloud.shiftr.io
# Note the "cloud" in the URL - this is critical!
MQTT_BROKER          = "khiet1111.cloud.shiftr.io"      # Fixed: added "cloud"
MQTT_PORT            = 1883
MQTT_TOPIC_BASE      = "/smart_home/"
MQTT_USER            = "khiet1111"                  # Token Key from Shiftr.io
MQTT_PASSWORD        = "khiet"                      # Token Secret from Shiftr.io
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
HAND_DETECTION_CONFIDENCE = 0.5    # Reduced from 0.6 for speed
HAND_TRACKING_CONFIDENCE  = 0.4    # Reduced from 0.5

GESTURE_HOLD_TIME    = 1.5   # Reduced from 2.0 for faster response
CONFIRM_HOLD_TIME    = 0.6   # Reduced from 0.8
CONFIRM_ENTRY_DELAY  = 0.6   # Reduced from 0.8

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
