"""
main.py — Smart Home Face + Gesture Control
============================================
Entry point. Ties together:
  FaceAuth       → identifies who is in front of camera
  GestureControl → reads hand gestures, gates on face auth
  MQTTHandler    → publishes commands to ESP32

Single camera, single window.
Run:  python3 main.py
"""

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

    # ── Single camera open — never re-opened ──────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    WIN = "Smart Home"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, face.mouse_callback)

    print("\n========================================")
    print("  Smart Home — Face + Gesture Control")
    print("========================================")
    print("  e = Enroll face    d = Delete face")
    print("  r = Relock         ESC = Quit")
    print("  (Name entry is ON-SCREEN — no typing in terminal)\n")

    fps_time = time.time()
    fps      = 0.0

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret:
            print("[WARN] Frame read failed — retrying...")
            time.sleep(0.05)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        # ── Face recognition ──────────────────────────────────────────
        # process_frame draws face landmarks, locked/unlocked overlay,
        # handles enroll/delete states, returns annotated frame
        frame = face.process_frame(raw.copy(), key)
        face.handle_key(key)

        # ── Gesture recognition ───────────────────────────────────────
        # process_frame draws hand landmarks + gesture UI on same frame
        # passes face_unlocked so gestures are gated on auth
        frame = gesture.process_frame(frame, mqtt, face.is_unlocked())

        # ── Shared UI overlays ────────────────────────────────────────
        face.draw_status_bar(frame)   # LOCKED / UNLOCKED bar at top
        face.draw_debug(frame)        # shape/cosine scores bottom-left
        gesture.draw_fps(frame, fps)  # FPS bottom-right

        # MQTT connection indicator (top-right corner)
        mqtt_col = (0,200,0) if mqtt.is_connected() else (0,0,200)
        mqtt_lbl = "MQTT OK" if mqtt.is_connected() else "MQTT OFF"
        cv2.putText(frame, mqtt_lbl,
                    (frame.shape[1]-120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, mqtt_col, 2)

        cv2.imshow(WIN, frame)

        # FPS
        now      = time.time()
        fps      = 1.0 / (now - fps_time + 1e-6)
        fps_time = now

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    mqtt.stop()
    print("\nProgram ended.")


if __name__ == '__main__':
    main()
