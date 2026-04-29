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
        if not ret:
            continue

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break

        # ── 2. Create the shared frame ────────────────────────────────
        # We pass this single object to both modules to "stack" the UI
        frame = raw.copy()

        # ── 3. Logic Staggering (The Secret to High FPS) ──────────────
        # We rotate the load so the Jetson CPU doesn't choke.
        # Frame 1: Face Math | Frame 2: Gesture Math | Frame 3: UI Only
        run_face = (frame_count % 3 == 0)
        run_gest = (frame_count % 2 == 0)

        # Update Face Auth
        # Note: We added the 'skip_inference' argument to your modules
        face.process_frame(frame, key, skip_inference=not run_face)

        # Update Gesture Control (Gated by face.is_unlocked())
        gesture.process_frame(frame, mqtt, face.is_unlocked(), skip_inference=not run_gest)

        # ── 4. Global Overlays ────────────────────────────────────────
        # These are usually small functions that don't need skipping
        face.draw_debug(frame)
        gesture.draw_fps(frame, fps)

        # MQTT Status Indicator (Top Right)
        mqtt_col = (0, 200, 0) if mqtt.is_connected() else (0, 0, 200)
        cv2.putText(frame, "MQTT OK" if mqtt.is_connected() else "MQTT OFF",
                    (frame.shape[1] - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mqtt_col, 2)

        # ── 5. Display ────────────────────────────────────────────────
        cv2.imshow(WIN, frame)

        # Calculate actual performance
        now = time.time()
        fps = 1.0 / (now - fps_time + 1e-6)
        fps_time = now

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mqtt.stop()

if __name__ == '__main__':
    main()
