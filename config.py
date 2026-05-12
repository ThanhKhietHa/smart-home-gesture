
import os

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
CAMERA_WIDTH  = 640   
CAMERA_HEIGHT = 360
CAMERA_FPS    = 30

FACE_PROCESS_EVERY_N_FRAMES_LOCKED   = 2
FACE_PROCESS_EVERY_N_FRAMES_UNLOCKED = 90

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

GESTURE_HOLD_TIME    = 1.5
CONFIRM_HOLD_TIME    = 0.6
CONFIRM_ENTRY_DELAY  = 0.6

=
GESTURE_COMMANDS = {

    "Open Palm":  ("lights", "on"),    # confirm: Thumb Up=on, Thumb Down=off
    "Peace Sign": ("door",   "toggle"),# confirm: Thumb Up only (toggle)
    "Pointing Up":("ac",     "on"),    # confirm: Thumb Up=on, Thumb Down=off
    "Thumb Up":   ("window", "on"),    # confirm: Thumb Up=up(on), Thumb Down=down(off)
}

DEVICE_HAS_ONOFF = {
    "lights": True,    # Thumb Up=on, Thumb Down=off
    "ac":     True,    # Thumb Up=on, Thumb Down=off
    "window": True,    # Thumb Up=on(roll up), Thumb Down=off(roll down)
    "door":   False,   # Thumb Up only (toggle) — no off state
}

DEVICE_INITIAL_STATES = {
    "lights": 0,
    "door":   0,
    "ac":     0,
    "window": 0,
}
