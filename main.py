import cv2
import time
import config
import numpy as np
from face_auth       import FaceAuth
from gesture_control import GestureControl
from mqtt_handler    import MQTTHandler

def main():
    # 1. Initialize modules
    mqtt    = MQTTHandler()
    face    = FaceAuth()
    gesture = GestureControl()

    # Use V4L2 for Jetson stability (JetPack 6+)
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {config.CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    WIN = "Smart Home"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, face.mouse_callback)

    fps_time    = time.time()
    frame_count = 0
    fps         = 0.0

    print("System Running... Press ESC to quit.")

    # 2. Main Loop
    while cap.isOpened():
        ret, raw = cap.read()
        if not ret or raw is None:
            continue

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break

        # Pass keys to the face module (Enroll/Delete/Relock)
        face.handle_key(key)

        # Work on a fresh copy of the frame
        frame = raw.copy()

        # STAGGERING LOGIC:
        # Frame 1: Run Face AI, Skip Gesture AI
        # Frame 2: Skip Face AI, Run Gesture AI
        # Frame 3: Skip both (Pure UI/Drawing frame)
        # This keeps the CPU from hitting 100% and lagging the video
        run_face = (frame_count % 3 == 1)
        run_gest = (frame_count % 3 == 2)

        # Process Face (UI bar always draws, even if inference is skipped)
        face.process_frame(frame, key, skip_inference=not run_face)

        # Process Gestures (Devices list always draws)
        gesture.process_frame(frame, mqtt, face.is_unlocked(), skip_inference=not run_gest)

        # Global Overlays
        face.draw_debug(frame)
        
        # FPS Calculation
        now = time.time()
        fps = 1.0 / (now - fps_time + 1e-6)
        fps_time = now
        gesture.draw_fps(frame, fps)

        # MQTT Connection Indicator
        m_col = (0, 200, 0) if mqtt.is_connected() else (0, 0, 200)
        cv2.putText(frame, "MQTT OK" if mqtt.is_connected() else "MQTT OFF",
                    (frame.shape[1]-120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_col, 2)

        cv2.imshow(WIN, frame)

    # 3. Cleanup
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    mqtt.stop()

if __name__ == '__main__':
    main()
