import cv2
import time
# python3 test.py
# Test 1: Camera only (no AI)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

start = time.time()
count = 0
while time.time() - start < 5:
    ret, frame = cap.read()
    if ret:
        count += 1
camera_fps = count / 5
print(f"Camera only (no AI): {camera_fps:.1f} FPS")
cap.release()

# Test 2: Run your actual code with the FPS print
# If camera_only shows 30 FPS but your code shows 10 FPS,
# the bottleneck is AI, not camera.
