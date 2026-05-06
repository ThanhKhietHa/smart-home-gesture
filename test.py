"""
test_threaded.py
Tests face + hand running in SEPARATE THREADS simultaneously
to see if threading gives real parallel speedup on Jetson.
Also tests LIVE_STREAM mode inference speed.
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

# ── Grab a real frame ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
time.sleep(0.5)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Cannot read camera")
    exit(1)

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
print(f"Test frame: {frame.shape[1]}x{frame.shape[0]}")

# ── Test 1: Face + Hand in separate threads (IMAGE mode) ──────────────
print("\n--- Test 1: Face + Hand in SEPARATE THREADS (IMAGE mode) ---")

face_opts = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.4,
)
hand_opts = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
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
tf.start(); th.start()
tf.join(); th.join()
elapsed = time.time() - t_start

avg_face = sum(face_times)/len(face_times)*1000
avg_hand = sum(hand_times)/len(hand_times)*1000
print(f"Face avg: {avg_face:.1f}ms")
print(f"Hand avg: {avg_hand:.1f}ms")
print(f"Total wall time for 30 frames each: {elapsed:.2f}s")
print(f"Effective FPS (threaded): {30/elapsed:.1f}")
print(f"Sequential would take: {30*(avg_face+avg_hand)/1000:.2f}s")

# ── Test 2: LIVE_STREAM mode ───────────────────────────────────────────
print("\n--- Test 2: LIVE_STREAM mode (face only) ---")

face_results = []
result_times = []

def face_callback(result, image, ts):
    result_times.append(time.time())

face_live_opts = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=face_callback,
    num_faces=1,
    min_face_detection_confidence=0.4,
)
face_live = vision.FaceLandmarker.create_from_options(face_live_opts)

N = 30
send_times = []
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

# ── Test 3: Camera real FPS ───────────────────────────────────────────
print("\n--- Test 3: Camera real FPS at 320x240 ---")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
N = 60
t0 = time.time()
for _ in range(N):
    cap.read()
elapsed = time.time() - t0
cap.release()
print(f"Camera FPS: {N/elapsed:.1f}")
print(f"Per frame: {elapsed/N*1000:.1f}ms")

# ── Summary ───────────────────────────────────────────────────────────
print("\n======= SUMMARY =======")
print(f"Camera limit:          {N/elapsed:.0f} FPS")
print(f"Face model alone:      {1000/avg_face:.0f} FPS")
print(f"Hand model alone:      {1000/avg_hand:.0f} FPS")
print(f"Threaded (both):       {30/elapsed_thread:.0f} FPS" if False else f"Threaded (both):       {30/elapsed:.0f} FPS (approx)")
print(f"LIVE_STREAM face:      {throughput:.0f} FPS")
print(f"\nConclusion:")
if throughput > 1000/avg_face * 1.3:
    print("  LIVE_STREAM is significantly faster — switch to it")
elif 30/elapsed > 1000/(avg_face+avg_hand) * 1.3:
    print("  Threading gives real parallel speedup — current approach is correct")
else:
    print("  Both models share CPU — threading limited by Python GIL")
    print("  Best option: run only ONE model at a time (state-based scheduling)")
