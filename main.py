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

    # Use V4L2 for Jetson stability (standard for JetPack 6+)
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {config.CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    WIN = "Smart Home"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, face.mouse_callback)

    fps_time    = time.time()
    fps         = 0.0
    frame_count = 0

    print("\n========================================")
    print("  Smart Home — System Running")
    print("========================================")
    print("  Press ESC to quit.\n")

    # ── 2. Main Loop ──────────────────────────────────────────────────
    while cap.isOpened():
        ret, raw = cap.read()
        if not ret or raw is None:
            continue

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC key
            break

        # Start with a fresh frame copy for this loop iteration
        frame = raw.copy()

        # STAGGERING LOGIC:
        # Frame 1: Run Face AI, Skip Gesture AI
        # Frame 2: Skip Face AI, Run Gesture AI
        # Frame 3: Skip both (Pure UI/Drawing frame)
        # This keeps the Jetson Orin Nano responsive on CPU
        run_face = (frame_count % 3 == 1)
        run_gest = (frame_count % 3 == 2)

        # Both modules now ALWAYS draw their UI, but only sometimes do math
        face.process_frame(frame, key, skip_inference=not run_face)
        gesture.process_frame(frame, mqtt, face.is_unlocked(), skip_inference=not run_gest)

        # ── 3. Overlays ───────────────────────────────────────────────
        face.draw_debug(frame)
        gesture.draw_fps(frame, fps)

        # MQTT Top-Right Indicator
        m_col = (0, 200, 0) if mqtt.is_connected() else (0, 0, 200)
        m_txt = "MQTT OK" if mqtt.is_connected() else "MQTT OFF"
        cv2.putText(frame, m_txt, (frame.shape[1]-120, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_col, 2)

        # ── 4. Display ────────────────────────────────────────────────
        cv2.imshow(WIN, frame)

        # FPS Calculation
        now = time.time()
        fps = 1.0 / (now - fps_time + 1e-6)
        fps_time = now

    # ── 5. Cleanup ────────────────────────────────────────────────────
    print("\nShutting down...")
    cap.release()
    cv2.destroyAllWindows()
    mqtt.stop()
    print("Program ended.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
