# test_mediapipe.py - Measure MediaPipe speed
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("Loading MediaPipe models...")

# Face model
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='models/face_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_facial_transformation_matrixes=False,
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

# Hand model
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

print("Capturing test frame...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

for i in range(5):
    cap.read()  # Warm up

ret, frame = cap.read()
if not ret:
    print("Camera error")
    cap.release()
    exit()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# Test face detection
print("\nTesting face detection...")
times = []
for i in range(30):
    start = time.time()
    result = face_detector.detect(img)
    times.append(time.time() - start)

face_avg = sum(times) / len(times)
print(f"Face detection: {face_avg*1000:.1f}ms per frame")
print(f"Face-only FPS: {1/face_avg:.1f}")

# Test hand detection
print("\nTesting hand detection...")
times = []
for i in range(30):
    start = time.time()
    result = hand_detector.detect(img)
    times.append(time.time() - start)

hand_avg = sum(times) / len(times)
print(f"Hand detection: {hand_avg*1000:.1f}ms per frame")
print(f"Hand-only FPS: {1/hand_avg:.1f}")

# Both combined
total_ms = (face_avg + hand_avg) * 1000
print(f"\nBOTH combined: {total_ms:.1f}ms per frame")
print(f"Max theoretical FPS: {1000/total_ms:.1f}")

cap.release()
