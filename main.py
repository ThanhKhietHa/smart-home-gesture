import cv2
import time
import config
import numpy as np
from face_auth       import FaceAuth
from gesture_control import GestureControl
from mqtt_handler    import MQTTHandler

def main():
    # ── 1. Init modules ───────────────────────────────────────────────
    mqtt    = MQTTHandler()
    face    = FaceAuth()
    gesture = GestureControl()

    # Use V4L2 for Jetson stability
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    WIN = "Smart Home"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, face.mouse_callback)

    fps_time    = time.time()
    fps         = 0.0
    frame_count = 0

    print("System Running... Press ESC to quit.")

while cap.isOpened():
        ret, raw = cap.read()
        if not ret: continue

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break

        # Start with a fresh frame copy for this loop iteration
        frame = raw.copy()

        # STAGGERING LOGIC:
        # Frame 1: Run Face, Skip Gesture
        # Frame 2: Skip Face, Run Gesture
        # Frame 3: Skip both (pure UI frame)
        run_face = (frame_count % 3 == 1)
        run_gest = (frame_count % 3 == 2)

        # Both modules now ALWAYS draw their UI, but only sometimes do math
        face.process_frame(frame, key, skip_inference=not run_face)
        gesture.process_frame(frame, mqtt, face.is_unlocked(), skip_inference=not run_gest)

        # Standard Overlays
        face.draw_debug(frame)
        gesture.draw_fps(frame, fps)

        # MQTT Top-Right Indicator
        m_col = (0, 200, 0) if mqtt.is_connected() else (0, 0, 200)
        cv2.putText(frame, "MQTT OK" if mqtt.is_connected() else "MQTT OFF",
                    (frame.shape[1]-120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_col, 2)

        cv2.imshow("Smart Home", frame)

        # FPS Calc
        now = time.time()
        fps = 1.0 / (now - fps_time + 1e-6)
        fps_time = now

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mqtt.stop()

if __name__ == '__main__':
    main()
