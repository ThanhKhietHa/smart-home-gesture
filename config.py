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
# CAMERA
# =====================================================================
# USB camera (Kisonli HD 1080) on Jetson / Windows
CAMERA_INDEX  = 0       # change to 1 if default camera is not the USB one
CAMERA_WIDTH  = 640     # lower = faster. 320x240 for max FPS on Jetson
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30

# =====================================================================
# MQTT
# =====================================================================
MQTT_BROKER         = "localhost"   # change to ESP32 broker IP if needed
MQTT_PORT           = 1883
MQTT_TOPIC_BASE     = "/smart_home/"
MQTT_RECONNECT_DELAY = 3.0

# =====================================================================
# FACE RECOGNITION
# =====================================================================
# Tuned for KHIET's debug panel readings:
#   KHIET:   Shape=0.067  Cosine=0.003
#   Stranger: Shape=0.144  Cosine=0.014
# Thresholds set midway between the two:
FACE_SHAPE_THRESHOLD    = 0.10   # midpoint: 0.067 < 0.10 < 0.144
FACE_IDENTITY_THRESHOLD = 0.008  # midpoint: 0.003 < 0.008 < 0.014

# If getting false rejects → raise by 0.01
# If getting false accepts → lower by 0.01

FACE_CONFIRM_FRAMES  = 5      # consecutive match frames to unlock (lower = faster)
FACE_RELOCK_FRAMES   = 25     # consecutive no-match frames to re-lock
FACE_ENROLL_TARGET   = 40     # good frames per enrollment
FACE_AUTH_TIMEOUT    = 300.0  # seconds before auto re-lock
FACE_MIN_HEIGHT_FRAC = 0.20   # face must fill 20% of frame height (relaxed for USB)

# =====================================================================
# GESTURE RECOGNITION
# =====================================================================
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE  = 0.5   # slightly relaxed for lower FPS

GESTURE_HOLD_TIME    = 2.0   # seconds hold before confirm screen
CONFIRM_HOLD_TIME    = 0.8   # seconds hold thumb to confirm
CONFIRM_ENTRY_DELAY  = 0.8   # stabilise delay after entering confirm

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
