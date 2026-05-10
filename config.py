

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

CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640  # explicitly listed in camera MJPG modes
CAMERA_HEIGHT = 320  # native resolution avoids YUYV fallback
CAMERA_FPS    = 30

FACE_PROCESS_EVERY_N_FRAMES_LOCKED    = 2

FACE_PROCESS_EVERY_N_FRAMES_UNLOCKED  = 90

FACE_DETECTION_CONFIDENCE = 0.35
FACE_PRESENCE_CONFIDENCE  = 0.35

MQTT_BROKER          = "khiet1111.cloud.shiftr.io"
MQTT_PORT            = 1883
MQTT_TOPIC_BASE      = "/smart_home/"
MQTT_USER            = "khiet1111"
MQTT_PASSWORD        = "khiet"
MQTT_RECONNECT_DELAY = 3.0

FACE_SHAPE_THRESHOLD    = 0.10
FACE_IDENTITY_THRESHOLD = 0.008

FACE_CONFIRM_FRAMES  = 5
FACE_RELOCK_FRAMES   = 25
FACE_ENROLL_TARGET   = 40
FACE_AUTH_TIMEOUT    = 300.0
FACE_MIN_HEIGHT_FRAC = 0.20

HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE  = 0.4

GESTURE_HOLD_TIME   = 1.5

ACTION_HOLD_TIME    = 0.8

MENU_ENTRY_DELAY    = 0.5

MENU_TIMEOUT        = 3.0

ENTRY_GESTURES = {
    "Open Palm":   "lights",
    "Peace Sign":  "door",
    "Pointing Up": "ac",
    "Thumb Up":    "window",   
}

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
    "lights": "Thumb UP = ON   |   Thumb DOWN = OFF",
    "door":   "Thumb UP = Toggle",
    "ac":     "Thumb UP = ON   |   Thumb DOWN = OFF",
    "window": "Thumb UP = Roll Up  |  Thumb DOWN = Roll Down",
}

DEVICE_INITIAL_STATES = {
    "lights": 0,
    "door":   0,
    "ac":     0,
    "window": "stopped",
}
