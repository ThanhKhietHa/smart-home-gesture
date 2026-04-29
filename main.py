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

    fps_time    = time.time()
    fps         = 0.0
    frame_count = 0

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret:
            print("[WARN] Frame read failed — retrying...")
            time.sleep(0.05)
            continue

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC = quit
            break

        # ── Face recognition (every 3rd frame — saves CPU) ────────────
        # process_frame draws landmarks, locked/unlocked overlay,
        # handles enroll/delete states, returns annotated frame
        if frame_count % 3 == 0:
            frame = face.process_frame(raw.copy(), key)
        else:
            frame = raw.copy()

        face.handle_key(key)

        # ── Gesture recognition (every 2nd frame) ─────────────────────
        # gated on face auth — gestures blocked when locked
        if frame_count % 2 == 0:
            frame = gesture.process_frame(frame, mqtt, face.is_unlocked())

        # ── Shared UI overlays ────────────────────────────────────────
        face.draw_status_bar(frame)   # LOCKED / UNLOCKED bar at top
        face.draw_debug(frame)        # shape/cosine scores bottom-left
        gesture.draw_fps(frame, fps)  # FPS counter bottom-right

        # MQTT connection indicator top-right
        mqtt_col = (0, 200, 0) if mqtt.is_connected() else (0, 0, 200)
        mqtt_lbl = "MQTT OK"  if mqtt.is_connected() else "MQTT OFF"
        cv2.putText(frame, mqtt_lbl,
                    (frame.shape[1] - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, mqtt_col, 2)

        cv2.imshow(WIN, frame)

        # FPS calculation
        now      = time.time()
        fps      = 1.0 / (now - fps_time + 1e-6)
        fps_time = now

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    mqtt.stop()
    print("\nProgram ended.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print("\n[CRASH]", e)
        traceback.print_exc()
        input("Press Enter to close...")
