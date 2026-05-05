# test_fps.py - Run this to diagnose FPS
import cv2
import time

print("Testing camera FPS...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Warm up
for i in range(10):
    ret, frame = cap.read()

# Test 100 frames
start = time.time()
frame_count = 0
while frame_count < 100:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
    else:
        print("Failed to read frame")
        break

elapsed = time.time() - start
camera_fps = frame_count / elapsed
print(f"Camera FPS (320x240): {camera_fps:.1f}")
print(f"Average time per frame: {elapsed/frame_count*1000:.1f}ms")

cap.release()
