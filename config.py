"""
config.py — Central configuration for Smart Home Gesture Control
================================================================
Change settings HERE only.
"""

import os

# =====================================================================
# PATHS
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FACE_MODEL_PATH = os.path.join(MODELS_DIR, 'face_landmarker.task')
HAND_MODEL_PATH = os.path.join(MODELS_DIR, 'hand_landmarker.task')

ENROLLED_FILE = os.path.join(DATA_DIR, 'enrolled_faces.pkl')
ENROLL_PHOTOS_DIR = os.path.join(DATA_DIR, 'enroll_photos')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ENROLL_PHOTOS_DIR, exist_ok=True)

# =====================================================================
# CAMERA
# =====================================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# =====================================================================
# MQTT
# =====================================================================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_BASE = "/smart_home/"
MQTT_RECONNECT_DELAY = 3.0

# =====================================================================
# FACE RECOGNITION — IMPROVED SETTINGS
# =====================================================================

# --- Thresholds ---
FACE_SHAPE_THRESHOLD = 0.12      # Slightly relaxed (was 0.10)
FACE_IDENTITY_THRESHOLD = 0.010  # Slightly relaxed (was 0.008)

# --- Enrollment Settings ---
FACE_ENROLL_TARGET = 25          # Reduced from 40 → much easier to complete
FACE_ENROLL_MAX_YAW = 48         # NEW: More forgiving yaw
FACE_ENROLL_MAX_PITCH = 38       # NEW: More forgiving pitch

# --- Recognition & Timing ---
FACE_CONFIRM_FRAMES = 6
FACE_RELOCK_FRAMES = 25          # Slightly reduced
FACE_AUTH_TIMEOUT = 300.0        # 5 minutes
FACE_MIN_HEIGHT_FRAC = 0.25

# --- Debug / Tuning Helper ---
# Set this to True while tuning, then set back to False
SHOW_DEBUG_PANEL = True

# =====================================================================
# GESTURE RECOGNITION
# =====================================================================
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE = 0.6

GESTURE_HOLD_TIME = 2.0
CONFIRM_HOLD_TIME = 0.8
CONFIRM_ENTRY_DELAY = 0.8

# =====================================================================
# DEVICE → GESTURE MAPPING
# =====================================================================
GESTURE_COMMANDS = {
    "Open Palm": ("lights", "on"),
    "Fist": ("lights", "off"),
    "Peace Sign": ("door", "toggle"),
    "Pointing Up": ("ac", "off"),
    "Pointing Down": ("ac", "on"),
    "Three Fingers": ("fan", "on"),
    "Four Fingers": ("fan", "off"),
    "Pinch": ("curtains", "open"),
    "Spread": ("curtains", "close"),
    "Thumb Up": ("window", "roll_up"),
    "Thumb Down": ("window", "roll_down"),
}

DEVICE_INITIAL_STATES = {
    "lights": 0,
    "fan": 0,
    "door": 0,
    "ac": 0,
    "curtains": "stopped",
    "window": "stopped",
}
