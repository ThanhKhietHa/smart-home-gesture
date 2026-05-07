"""
config.py - Configuration for Gesture Control System
New mapping based on your requirements
"""

import os

# =====================================================================
# PATHS
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

# =====================================================================
# MEDIAPIPE CONFIDENCE
# =====================================================================
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5

# =====================================================================
# GESTURE TIMING
# =====================================================================
GESTURE_HOLD_TIME = 1.0          # Seconds to hold for activation
CONFIRM_HOLD_TIME = 0.8          # Seconds to hold thumb for confirmation
CONFIRM_ENTRY_DELAY = 0.3        # Delay before showing confirmation

# =====================================================================
# NEW GESTURE TO ACTION MAPPING
# =====================================================================
GESTURE_COMMANDS = {
    # Light control
    "Thumb Up":    ("light", "on"),      # Thumb up = turn light ON
    "Thumb Down":  ("light", "off"),     # Thumb down = turn light OFF
    
    # Door control  
    "Peace Sign":  ("door", "toggle"),    # Peace sign = toggle door
    
    # AC control
    "Pointing Up": ("ac", "toggle"),      # Pointing up = toggle AC
    
    # Window control (entered via Thumb Up menu)
    "Window Open":   ("window", "open"),   # Thumb up in window mode = open
    "Window Close":  ("window", "close"),  # Thumb down in window mode = close
    
    # Universal Cancel
    "Open Palm":   ("cancel", "cancel"),   # Open palm = cancel any action
}

# =====================================================================
# DEVICE STATES (for display)
# =====================================================================
DEVICE_INITIAL_STATES = {
    "light": 0,      # 0 = off, 1 = on
    "door": "closed", # closed/open
    "ac": 0,         # 0 = off, 1 = on
    "window": "closed", # closed/open
}

# =====================================================================
# WINDOW CONTROL MODE (entered via Thumb Up)
# =====================================================================
WINDOW_CONTROL_MODE = {
    "enabled": False,
    "entry_gesture": "Thumb Up",
    "open_gesture": "Thumb Up",    # In window mode, Thumb Up = open
    "close_gesture": "Thumb Down",  # In window mode, Thumb Down = close
    "cancel_gesture": "Open Palm",  # Exit window mode
}
