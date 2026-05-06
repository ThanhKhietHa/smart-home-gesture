"""
test_threaded.py
Tests face + hand running in SEPARATE THREADS simultaneously
to see if threading gives real parallel speedup on Jetson.
Also tests camera resolutions and LIVE_STREAM mode.
Run: python3 test_threaded.py
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import threading
import numpy as np

FACE_MODEL = 'models/face_landmarker.task'
HAND_MODEL = 'models/hand_landmarker.task'

# =====================================================================
# TEST 0: Camera Resolution Test (Find best working resolution)
# =====================================================================
print("\n=== TEST 0: Camera Resolution Test ===")
print("Testing which resolutions your Kisonli camera actually supports...\n")

resolutions = [
    (320, 240),   # Your desired resolution
    (424, 240),   # Common 16:9 width at 240p
    (640, 240),   # Wide but short
    (640, 360),   # 360p - likely default
    (640, 480),   # 480p
    (480, 360),   # Alternative
    (416, 240),   # Another try
]

best_fps = 0
best_res = None
best_actual = None

for w, h in resolutions:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"  Can't open camera for {w}x{h}")
        continue
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Warm up
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            break
    
    if not ret:
        print(f"  {w}x{h} -> Camera read failed")
        cap.release()
        continue
    
    actual_h, actual_w = frame.shape[:2]
    actual = f"{actual_w}x{actual_h}"
    
    # Test FPS
    start = time.time()
    frame_count = 0
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            frame_count += 1
    elapsed = time.time() - start
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    cap.release()
    
    status = "✅" if fps > 15 else "⚠️" if fps > 10 else "❌"
    print(f"  {status} Request {w}x{h} -> Actual {actual} -> {fps:.1f} FPS")
    
    if fps > best_fps:
        best_fps = fps
        best_res = (w, h)
        best_actual = actual

print(f"\n🏆 BEST RESOLUTION: {best_res} (actual: {best_actual}) at {best_fps:.1f} FPS")

# Use the best resolution for the rest of the tests
USE_RESOLUTION = best_res if best_res else (320, 240)
print(f"Using resolution: {USE_RESOLUTION[0]}x{USE_RESOLUTION[1]}\n")

# =====================================================================
# Grab test frame at the best resolution
# =====================================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, USE_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USE_RESOLUTION[1])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
time.sleep(0.5)

# Read multiple frames to get stable image
for _ in range(5):
    ret, frame = cap.read()
    
if not ret:
    print("ERROR: Cannot read camera")
    cap.release()
    exit(1)

actual_h, actual_w = frame.shape[:2]
print(f"Test frame captured: {actual_w}x{actual_h}")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
cap.release()

# =====================================================================
# TEST 1: Face + Hand in separate threads (IMAGE mode)
# =====================================================================
print("\n=== TEST 1: Face + Hand in SEPARATE THREADS (IMAGE mode) ===")

face_opts = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.35,  # Lower for speed
    output_facial_transformation_matrixes=False,  # Speed boost
)
hand_opts = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.4,
)
face_lm = vision.FaceLandmarker.create_from_options(face_opts)
hand_lm = vision.HandLandmarker.create_from_options(hand_opts)

face_times = []
hand_times = []

def run_face(n=30):
    for _ in range(n):
        t = time.time()
        face_lm.detect(mp_img)
        face_times.append(time.time() - t)

def run_hand(n=30):
    for _ in range(n):
        t = time.time()
        hand_lm.detect(mp_img)
        hand_times.append(time.time() - t)

t_start = time.time()
tf = threading.Thread(target=run_face, args=(30,))
th = threading.Thread(target=run_hand, args=(30,))
tf.start()
th.start()
tf.join()
th.join()
elapsed = time.time() - t_start

avg_face = sum(face_times)/len(face_times)*1000
avg_hand = sum(hand_times)/len(hand_times)*1000
print(f"Face avg: {avg_face:.1f}ms")
print(f"Hand avg: {avg_hand:.1f}ms")
print(f"Total wall time for 30 frames each: {elapsed:.2f}s")
print(f"Effective FPS (threaded): {30/elapsed:.1f}")
print(f"Sequential would take: {30*(avg_face+avg_hand)/1000:.2f}s")
print(f"Speedup factor: {(avg_face+avg_hand)/(elapsed/30*1000):.1f}x")

# =====================================================================
# TEST 2: LIVE_STREAM mode
# =====================================================================
print("\n=== TEST 2: LIVE_STREAM mode (face only) ===")

face_results = []
result_times = []
send_times = []

def face_callback(result, image, ts):
    result_times.append(time.time())

face_live_opts = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=face_callback,
    num_faces=1,
    min_face_detection_confidence=0.35,
    output_facial_transformation_matrixes=False,
)
face_live = vision.FaceLandmarker.create_from_options(face_live_opts)

N = 30
ts = 0
for i in range(N):
    ts += 33  # ~30fps timestamps
    t = time.time()
    face_live.detect_async(mp_img, ts)
    send_times.append(time.time() - t)
    time.sleep(0.033)

time.sleep(0.5)  # wait for callbacks

avg_send = sum(send_times)/len(send_times)*1000
throughput = len(result_times) / (result_times[-1]-result_times[0]) if len(result_times)>1 else 0
print(f"detect_async call time: {avg_send:.2f}ms (non-blocking)")
print(f"Callbacks received: {len(result_times)}/{N}")
print(f"Effective throughput: {throughput:.1f} FPS")

# =====================================================================
# TEST 3: Camera real FPS at best resolution
# =====================================================================
print(f"\n=== TEST 3: Camera real FPS at {USE_RESOLUTION[0]}x{USE_RESOLUTION[1]} ===")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, USE_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USE_RESOLUTION[1])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Warm up
for _ in range(5):
    cap.read()

N = 60
t0 = time.time()
frames_read = 0
for _ in range(N):
    ret, frame = cap.read()
    if ret:
        frames_read += 1
elapsed = time.time() - t0
cap.release()

camera_fps = frames_read / elapsed
print(f"Camera FPS: {camera_fps:.1f}")
print(f"Per frame: {elapsed/frames_read*1000:.1f}ms")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "="*50)
print("SUMMARY")
print("="*50)

# Calculate sequential FPS
sequential_fps = 1000 / (avg_face + avg_hand)

print(f"Camera max FPS:          {camera_fps:.0f} FPS")
print(f"Face model alone:        {1000/avg_face:.0f} FPS")
print(f"Hand model alone:        {1000/avg_hand:.0f} FPS")
print(f"Sequential (face+hand):  {sequential_fps:.0f} FPS")
print(f"Threaded (face+hand):    {30/elapsed:.0f} FPS")
print(f"LIVE_STREAM face:        {throughput:.0f} FPS")

print("\n" + "="*50)
print("RECOMMENDATION")
print("="*50)

# Find bottleneck
bottleneck = max(avg_face, avg_hand)
if camera_fps < 15:
    print(f"⚠️  CAMERA is bottleneck: {camera_fps:.1f} FPS")
    print(f"   → Need lower resolution or different camera")
elif sequential_fps < 12:
    print(f"⚠️  MediaPipe is bottleneck: {sequential_fps:.1f} FPS")
    print(f"   → Need frame skipping or better scheduling")
elif 30/elapsed > sequential_fps * 1.2:
    print(f"✅ THREADING works! Speedup: {(30/elapsed)/sequential_fps:.1f}x")
    print(f"   → Keep current architecture")
else:
    print(f"⚠️  Python GIL limiting parallelism")
    print(f"   → Run only ONE model at a time based on state")

# Best config for your hardware
print(f"\n🎯 OPTIMAL CONFIGURATION for your Jetson:")
print(f"   Resolution: {USE_RESOLUTION[0]}x{USE_RESOLUTION[1]}")
print(f"   Face process every: {max(1, int(32/bottleneck))} frames")
print(f"   Expected FPS: {min(camera_fps, 1000/(bottleneck/2)):.0f}")

# Save results to file
with open('camera_test_results.txt', 'w') as f:
    f.write(f"Best resolution: {USE_RESOLUTION[0]}x{USE_RESOLUTION[1]}\n")
    f.write(f"Best FPS: {camera_fps:.1f}\n")
    f.write(f"Face time: {avg_face:.1f}ms\n")
    f.write(f"Hand time: {avg_hand:.1f}ms\n")
    f.write(f"Threaded FPS: {30/elapsed:.1f}\n")
    
print("\nResults saved to camera_test_results.txt")
