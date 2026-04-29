import cv2
import time
import config
from face_auth       import FaceAuth
from gesture_control import GestureControl
from mqtt_handler    import MQTTHandler

def main():
    # ── Init modules ──────────────────────────────────────────────────
    mqtt    = MQTTHandler()
    face    = FaceAuth()
    gesture = GestureControl()

    # Use V4L2 backend for better stability on Jetson
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

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret:
            continue

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break

        # ── 1. Create a single 'Canvas' for this frame ────────────────
        # We modify this 'frame' object throughout the loop
        frame = raw.copy()

        # ── 2. Face Recognition Logic ─────────────────────────────────
        face.handle_key(key)
        
        # We only run the heavy AI math every 3rd frame
        # BUT we let the function draw its UI every single frame
        do_face_math = (frame_count % 3 == 0)
        # Note: You may need to tweak face_auth.py to accept 'do_math' 
        # or simply rely on this logic:
        if do_face_math:
            face.process_frame(frame, key)
        else:
            # If skipping math, still draw the existing status/UI
            face.draw_status_bar(frame)

        # ── 3. Gesture Recognition Logic ──────────────────────────────
        # Only attempt gestures if face is unlocked
        is_unlocked = face.is_unlocked()
        
        # Run gesture math every 2nd frame to keep FPS up
        do_gesture_math = (frame_count % 2 == 0)
        
        if is_unlocked:
            if do_gesture_math:
                gesture.process_frame(frame, mqtt, True)
            else:
                # Keep drawing the device status list so it doesn't flicker
                gesture._draw_devices(frame) 
        else:
            # If locked, draw the "Blocked" UI or device list
            gesture._draw_devices(frame)

        # ── 4. Global UI Overlays (Always Draw) ────────────────────────
        face.draw_debug(frame)
        gesture.draw_fps(frame, fps)

        # MQTT Indicator
        mqtt_col = (0, 200, 0) if mqtt.is_connected() else (0, 0, 200)
        cv2.putText(frame, "MQTT OK" if mqtt.is_connected() else "MQTT OFF",
                    (frame.shape[1] - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, mqtt_col, 2)

        cv2.imshow(WIN, frame)

        # FPS calculation
        now = time.time()
        fps = 1.0 / (now - fps_time + 1e-6)
        fps_time = now

    cap.release()
    cv2.destroyAllWindows()
    mqtt.stop()

if __name__ == '__main__':
    main()
