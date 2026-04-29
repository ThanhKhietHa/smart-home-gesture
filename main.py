import cv2
import time
import config
import numpy as np
from face_auth import FaceAuth
from gesture_control import GestureControl
from mqtt_handler import MQTTHandler


def main():
    # 1. Initialize modules
    mqtt = MQTTHandler()
    face = FaceAuth()
    gesture = GestureControl()

    # Camera setup (V4L2 recommended for Jetson)
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {config.CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)          # Optional: request 30fps

    WIN = "Smart Home"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, face.mouse_callback)

    print("System Running... Press ESC to quit.")

    frame_count = 0
    fps = 0.0
    last_fps_time = time.time()
    fps_accum = 0.0
    fps_count = 0

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret or raw is None:
            print("[WARN] Failed to read frame")
            continue

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        # Make a working copy
        frame = raw.copy()

        # ── Staggered Inference ─────────────────────────────────────
        run_face = (frame_count % 3 == 1)
        run_gest = (frame_count % 3 == 2)

        # Always handle keys in RECOGNISE state
        if face._state == "RECOGNISE":        # or expose a method is_in_recognise()
            face.handle_key(key)

        # Process Face Recognition + UI
        face.process_frame(frame, key, skip_inference=not run_face)

        # Process Gestures (only when unlocked in most cases)
        gesture.process_frame(frame, mqtt, face.is_unlocked(), skip_inference=not run_gest)

        # ── Final Drawing Layer ─────────────────────────────────────
        face.draw_debug(frame)                    # Debug info at bottom
        gesture.draw_fps(frame, fps)              # Make sure this exists in gesture_control

        # MQTT status
        m_col = (0, 200, 0) if mqtt.is_connected() else (0, 0, 200)
        cv2.putText(frame, "MQTT OK" if mqtt.is_connected() else "MQTT OFF",
                    (frame.shape[1] - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_col, 2)

        # ── FPS Calculation (Smoothed) ───────────────────────────────
        fps_count += 1
        fps_accum += 1.0 / (time.time() - last_fps_time + 1e-6)
        last_fps_time = time.time()

        if fps_count >= 15:                       # Update FPS every 15 frames
            fps = fps_accum / fps_count
            fps_accum = 0.0
            fps_count = 0

        cv2.imshow(WIN, frame)

    # 3. Cleanup
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    mqtt.stop()


if __name__ == '__main__':
    main()
