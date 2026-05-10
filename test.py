#!/usr/bin/env python3
# python3 test.py
"""
Bottleneck Diagnostic Tool
Tests each component to find the real FPS limit
"""

import cv2
import time
import numpy as np
import threading
import config

# Import MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("=" * 60)
print("BOTTLENECK DIAGNOSTIC TOOL")
print("=" * 60)

# ------------------------------------------------------------------
# Helper function to initialize camera
# ------------------------------------------------------------------
def init_camera(mjpg=True, width=640, height=360):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if mjpg:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Verify settings
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Camera: {actual_w}x{actual_h}, Format: {fourcc_str}, FPS: {actual_fps:.0f}")
    
    return cap

# ------------------------------------------------------------------
# TEST 1: Camera only (absolute max)
# ------------------------------------------------------------------
print("\n[TEST 1] Camera ONLY (no processing)")
cap = init_camera(mjpg=True, width=640, height=360)

for _ in range(10):
    cap.read()

start = time.time()
count = 0
while time.time() - start < 3.0:
    ret, frame = cap.read()
    if ret:
        count += 1
camera_fps = count / 3.0
print(f"  ▶ RESULT: {camera_fps:.1f} FPS\n")
cap.release()

# ------------------------------------------------------------------
# TEST 2: Camera + imshow only (rendering test)
# ------------------------------------------------------------------
print("[TEST 2] Camera + imshow (rendering test)")
cap = init_camera(mjpg=True, width=640, height=360)

for _ in range(10):
    cap.read()

cv2.namedWindow("test", cv2.WINDOW_NORMAL)

start = time.time()
count = 0
while time.time() - start < 3.0:
    ret, frame = cap.read()
    if ret:
        count += 1
        cv2.imshow("test", frame)
        cv2.waitKey(1)
render_fps = count / 3.0
print(f"  ▶ RESULT: {render_fps:.1f} FPS\n")

cv2.destroyAllWindows()
cap.release()

# ------------------------------------------------------------------
# TEST 3: MediaPipe Hand (CPU + inference only, no rendering)
# ------------------------------------------------------------------
print("[TEST 3] MediaPipe Hand ONLY (no rendering)")
cap = init_camera(mjpg=True, width=640, height=360)

# Options for testing
MODEL_COMPLEXITY_OPTIONS = {
    "heavy": 1,  # model_complexity=1 (slower)
    "lite": 0,   # model_complexity=0 (faster)
}

for complexity_name, complexity_value in MODEL_COMPLEXITY_OPTIONS.items():
    print(f"\n  Testing model_complexity={complexity_value} ({complexity_name})...")
    
    # Re-initialize camera
    cap = init_camera(mjpg=True, width=640, height=360)
    
    # Create hand landmarker
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=config.HAND_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    
    # Warm up
    ret, test_frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        for _ in range(5):
            hand_landmarker.detect(img)
    
    # Measure FPS
    start = time.time()
    count = 0
    frames_processed = 0
    
    while time.time() - start < 3.0:
        ret, frame = cap.read()
        if not ret:
            continue
        count += 1
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hand_landmarker.detect(img)
        frames_processed += 1
    
    hand_fps = count / 3.0
    print(f"  ▶ Result: {hand_fps:.1f} FPS ({frames_processed} frames processed)")
    cap.release()
    hand_landmarker.close()

# ------------------------------------------------------------------
# TEST 4: MediaPipe Hand WITH rendering (draw landmarks + imshow)
# ------------------------------------------------------------------
print("\n[TEST 4] MediaPipe Hand WITH rendering (drawing + imshow)")
cap = init_camera(mjpg=True, width=640, height=360)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # Lighter model
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
)

for _ in range(10):
    cap.read()

cv2.namedWindow("test", cv2.WINDOW_NORMAL)

start = time.time()
count = 0
while time.time() - start < 3.0:
    ret, frame = cap.read()
    if not ret:
        continue
    count += 1
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("test", frame)
    cv2.waitKey(1)

render_fps = count / 3.0
print(f"  ▶ RESULT: {render_fps:.1f} FPS")

cv2.destroyAllWindows()
cap.release()
hands.close()

# ------------------------------------------------------------------
# TEST 5: Threaded capture test (bypassing OpenCV blocking)
# ------------------------------------------------------------------
print("\n[TEST 5] Threaded capture (non-blocking camera read)")

class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
    
    def write(self, frame):
        with self.lock:
            self.frame = frame
    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

def camera_thread_func(cap, buffer, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            buffer.write(frame)

cap = init_camera(mjpg=True, width=640, height=360)
buffer = FrameBuffer()
stop_event = threading.Event()

thread = threading.Thread(target=camera_thread_func, args=(cap, buffer, stop_event))
thread.start()

time.sleep(0.5)

start = time.time()
count = 0
while time.time() - start < 3.0:
    frame = buffer.read()
    if frame is not None:
        count += 1

stop_event.set()
thread.join(timeout=1)
threaded_fps = count / 3.0
print(f"  ▶ RESULT: {threaded_fps:.1f} FPS")

cap.release()

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY - Where is the bottleneck?")
print("=" * 60)
print(f"  Camera only:           {camera_fps:.1f} FPS")
print(f"  Camera + imshow:       {render_fps:.1f} FPS")
print(f"  Camera + Hand (lite):  {hand_fps:.1f} FPS")
print(f"  Camera + Hand (heavy): {hand_fps:.1f} FPS")
print(f"  Threaded capture:      {threaded_fps:.1f} FPS")
print("=" * 60)

print("\n📊 INTERPRETATION:")
print("-" * 40)

if camera_fps < 15:
    print("  ❌ Camera is TOO SLOW (<15 FPS)")
    print("     → Check USB port, try different resolution, or buy new camera")
elif render_fps < camera_fps * 0.8:
    print("  ❌ imshow() is the bottleneck (rendering on Jetson is slow)")
    print("     → Reduce display resolution or comment out imshow")
elif hand_fps < 15:
    print("  ❌ MediaPipe inference is the bottleneck")
    print("     → Use model_complexity=0, num_hands=1")
    print("     → Download float16 hand model")
    print("     → Reduce GESTURE_PROCESS_EVERY_N_FRAMES")
else:
    print("  ✅ No single bottleneck found")
    print("     → Check CPU governor: sudo nvpmodel -m 0 && sudo jetson_clocks")
    print("     → Check temperature: cat /sys/devices/virtual/thermal/thermal_zone*/temp")

print("\n✅ Done")
