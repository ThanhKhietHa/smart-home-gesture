import cv2
import time
import config
import numpy as np
from face_auth       import FaceAuth
from gesture_control import GestureControl
from mqtt_handler    import MQTTHandler

def main():
    mqtt = MQTTHandler()
    face = FaceAuth()
    gesture = GestureControl()

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened(): return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    fps_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret or raw is None: continue

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break

        frame = raw.copy()

        # STAGGERING: Rotate the CPU load
        # Frame 1: Face AI | Frame 2: Gesture AI | Frame 3: UI-only
        run_face = (frame_count % 3 == 1)
        run_gest = (frame_count % 3 == 2)

        # Both modules now accept 'skip_inference'
        face.process_frame(frame, key, skip_inference=not run_face)
        gesture.process_frame(frame, mqtt, face.is_unlocked(), skip_inference=not run_gest)

        # Draw Global UI
        face.draw_debug(frame)
        gesture.draw_fps(frame, 1.0 / (time.time() - fps_time + 1e-6))
        fps_time = time.time()

        cv2.imshow("Smart Home", frame)

    cap.release()
    cv2.destroyAllWindows()
    mqtt.stop()

if __name__ == '__main__':
    main()
