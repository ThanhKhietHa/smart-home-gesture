#!/usr/bin/env python3
"""Diagnose where the FPS bottleneck really is"""
#python3 test.py
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

print("=" * 50)
print("FPS DIAGNOSTIC TOOL")
print("=" * 50)

# ------------------------------------------------------------------
# TEST 1: Camera only (no AI at all)
# ------------------------------------------------------------------
print("\n[TEST 1] Camera only - no AI...")
cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)

# Try to set MJPG
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Warm up
for _ in range(10):
    cap.read()

start = time.time()
frame_count = 0
while time.time() - start < 3.0:
    ret, frame = cap.read()
    if ret:
        frame_count += 1

camera_fps = frame_count / 3.0
print(f"     Result: {camera_fps:.1f} FPS")

# ------------------------------------------------------------------
# TEST 2: Just reading from buffer (simulating display thread)
# ------------------------------------------------------------------
print("\n[TEST 2] Buffer read + display loop only...")

class SimpleBuffer:
    def __init__(self):
        self.frame = None
    def write(self, frame):
        self.frame = frame
    def read(self):
        return self.frame

buf = SimpleBuffer()
stop = False

def camera_reader():
    global stop
    while not stop:
        ret, frame = cap.read()
        if ret:
            buf.write(frame)

import threading
reader_thread = threading.Thread(target=camera_reader, daemon=True)
reader_thread.start()

start = time.time()
frame_count = 0
while time.time() - start < 3.0:
    frame = buf.read()
    if frame is not None:
        frame_count += 1

stop = True
reader_thread.join(timeout=1)
buffer_fps = frame_count / 3.0
print(f"     Result: {buffer_fps:.1f} FPS")

# ------------------------------------------------------------------
# TEST 3: Hand model only (no face, no threads)
# ------------------------------------------------------------------
print("\n[TEST 3] Hand model ONLY (sequential, no threading)...")

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.HAND_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Get a test frame
ret, test_frame = cap.read()
if not ret:
    print("     ERROR: Cannot get test frame")
    test_frame = np.zeros((360, 640, 3), dtype=np.uint8)

rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# Warm up
for _ in range(5):
    hand_landmarker.detect(img)

start = time.time()
iterations = 20
for _ in range(iterations):
    hand_landmarker.detect(img)
elapsed = time.time() - start
hand_ms = (elapsed / iterations) * 1000
hand_fps = 1000 / hand_ms
print(f"     Hand model: {hand_ms:.1f}ms per frame → {hand_fps:.1f} FPS max")

# ------------------------------------------------------------------
# TEST 4: Face model only (no hand, no threads)
# ------------------------------------------------------------------
print("\n[TEST 4] Face model ONLY (sequential, no threading)...")

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# Warm up
for _ in range(5):
    face_landmarker.detect(img)

start = time.time()
for _ in range(iterations):
    face_landmarker.detect(img)
elapsed = time.time() - start
face_ms = (elapsed / iterations) * 1000
face_fps = 1000 / face_ms
print(f"     Face model: {face_ms:.1f}ms per frame → {face_fps:.1f} FPS max")

# ------------------------------------------------------------------
# TEST 5: Both models sequential (what your 3-thread design avoids)
# ------------------------------------------------------------------
print("\n[TEST 5] Both models SEQUENTIAL (worst case)...")

start = time.time()
for _ in range(iterations):
    hand_landmarker.detect(img)
    face_landmarker.detect(img)
elapsed = time.time() - start
both_ms = (elapsed / iterations) * 1000
both_fps = 1000 / both_ms
print(f"     Both sequential: {both_ms:.1f}ms per cycle → {both_fps:.1f} FPS max")

# ------------------------------------------------------------------
# THEORETICAL MAX WITH MULTI-THREADING
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("THEORETICAL MAX FPS (with perfect threading)")
print("=" * 50)
print(f"  Camera max:        {camera_fps:.1f} FPS (hardware limit)")
print(f"  Hand model max:    {hand_fps:.1f} FPS")
print(f"  Face model max:    {face_fps:.1f} FPS")
print(f"  Overlap possible:  {hand_fps + face_fps:.1f} FPS (if both run in parallel)")
print(f"  YOUR REAL LIMIT:   {min(camera_fps, hand_fps + face_fps):.1f} FPS")
print("=" * 50)

# Clean up
cap.release()
hand_landmarker.close()
face_landmarker.close()
