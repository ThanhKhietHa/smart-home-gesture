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
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 360
CAMERA_FPS    = 30

# =====================================================================
# PERFORMANCE OPTIMIZATIONS
# =====================================================================
FACE_PROCESS_EVERY_N_FRAMES_LOCKED = 2
FACE_PROCESS_EVERY_N_FRAMES_UNLOCKED = 90
GESTURE_PROCESS_EVERY_N_FRAMES = 2

FACE_DETECTION_CONFIDENCE = 0.35
FACE_PRESENCE_CONFIDENCE = 0.35

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

GESTURE_HOLD_TIME    = 1.5
CONFIRM_HOLD_TIME    = 0.6
CONFIRM_ENTRY_DELAY  = 0.6

# =====================================================================
# DEVICE → GESTURE MAPPING (Updated - Removed fan/curtains gestures)
# =====================================================================
GESTURE_COMMANDS = {
    "Open Palm":     ("lights",   "toggle"),
    "Peace Sign":    ("door",     "toggle"),
    "Pointing Up":   ("ac",       "toggle"),
    "Thumb Up":      ("window",   "toggle"),
}

DEVICE_INITIAL_STATES = {
    "lights":   0,
    "door":     0,
    "ac":       0,
    "window":   0,
}
