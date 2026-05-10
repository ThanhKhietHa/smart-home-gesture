"""
main.py — Smart Home Face + Gesture Control
============================================
Architecture: 3 threads, NO inference mutex needed

  Thread 1 (main)    — camera read + display, never blocks
  Thread 2 (face)    — face auth; UNLOCKED=Haar ~3ms, LOCKED=MediaPipe ~30ms
  Thread 3 (gesture) — hand gesture MediaPipe ~45ms, only runs when UNLOCKED

Why no mutex needed:
  UNLOCKED: face thread uses Haar (OpenCV, not MediaPipe) → no conflict
  LOCKED:   gesture thread sleeps → no conflict
  The two MediaPipe models never run at the same time.

Expected FPS:
  LOCKED:   face MediaPipe every 5 frames, gesture sleeps → ~25 FPS display
  UNLOCKED: face Haar ~3ms + gesture MediaPipe ~45ms, independent threads
            → display thread runs at camera FPS, gesture updates ~20/s
"""

import cv2
import time
import threading
import config
from face_auth       import FaceAuth
from gesture_control import GestureControl
from mqtt_handler    import MQTTHandler


# =====================================================================
# THREAD-SAFE FRAME BUFFER
# =====================================================================
class FrameBuffer:
    def __init__(self):
        self._lock        = threading.Lock()
        self._raw         = None
        self._face_out    = None
        self._gesture_out = None

    def write_raw(self, frame):
        with self._lock: self._raw = frame.copy()

    def read_raw(self):
        with self._lock:
            return self._raw.copy() if self._raw is not None else None

    def write_face(self, frame):
        with self._lock: self._face_out = frame.copy()

    def read_face(self):
        with self._lock:
            return self._face_out.copy() if self._face_out is not None else None

    def write_gesture(self, frame):
        with self._lock: self._gesture_out = frame.copy()

    def read_gesture(self):
        with self._lock:
            return self._gesture_out.copy() if self._gesture_out is not None else None


# =====================================================================
# SHARED STATE
# =====================================================================
class SharedState:
    def __init__(self):
        self._lock          = threading.Lock()
        self._unlocked      = False
        self._name          = ""
        self._key           = -1
        self.show_feedback  = False
        self.feedback_msg   = ""
        self.feedback_color = (0, 255, 0)
        self.feedback_until = 0.0

    def set_auth(self, unlocked, name):
        with self._lock: self._unlocked = unlocked; self._name = name

    def is_unlocked(self):
        with self._lock: return self._unlocked

    def set_key(self, key):
        with self._lock: self._key = key

    def get_key(self):
        with self._lock:
            k = self._key; self._key = -1; return k

    def set_feedback(self, msg, color=(0,255,0), duration=1.5):
        with self._lock:
            self.show_feedback  = True
            self.feedback_msg   = msg
            self.feedback_color = color
            self.feedback_until = time.time() + duration

    def get_feedback(self):
        with self._lock:
            if self.show_feedback and time.time() < self.feedback_until:
                return self.feedback_msg, self.feedback_color
            self.show_feedback = False
            return None, None


# =====================================================================
# FACE THREAD
# =====================================================================
def face_thread(face, buf, state, stop_event):
    """
    UNLOCKED → process_presence_only() via Haar ~3ms every frame.
               face thread costs almost nothing, never blocks gesture.
    LOCKED   → process_frame() runs full MediaPipe every N frames.
               gesture thread is sleeping anyway, so no conflict.

    Always: write raw first so display never stalls, then overwrite
    with annotated result after processing.
    """
    while not stop_event.is_set():
        raw = buf.read_raw()
        if raw is None:
            time.sleep(0.008)
            continue

        # Push raw immediately so display thread never stalls
        buf.write_face(raw)

        key = state.get_key()

        # process_frame() internally routes:
        #   unlocked → Haar only (fast)
        #   locked   → full MediaPipe every N frames
        work      = raw.copy()
        annotated = face.process_frame(work, key)
        face.handle_key(key)
        state.set_auth(face.is_unlocked(), face.unlocked_name())

        buf.write_face(annotated)


# =====================================================================
# GESTURE THREAD
# =====================================================================
def gesture_thread(gesture, buf, state, mqtt, stop_event):
    """
    LOCKED:   sleeps — saves ~45ms per frame, face thread runs faster.
    UNLOCKED: runs hand MediaPipe every GESTURE_PROCESS_EVERY_N_FRAMES.
              No mutex needed — face thread is using Haar (not MediaPipe).
    """
    while not stop_event.is_set():
        if not state.is_unlocked():
            time.sleep(0.020)
            continue

        base = buf.read_face()
        if base is None:
            base = buf.read_raw()
        if base is None:
            time.sleep(0.008)
            continue

        frame, feedback = gesture.process_frame(base, mqtt, state.is_unlocked())
        if feedback:
            state.set_feedback(feedback[0], feedback[1])
        buf.write_gesture(frame)


# =====================================================================
# MAIN
# =====================================================================
def main():
    mqtt    = MQTTHandler()
    face    = FaceAuth()
    gesture = GestureControl()
    buf     = FrameBuffer()
    state   = SharedState()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)

    print("[MAIN] Camera warming up...")
    for _ in range(10):
        cap.read()

    WIN = "Smart Home"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, face.mouse_callback)

    print("\n========================================")
    print("  Smart Home — Face + Gesture Control  ")
    print("========================================")
    print("  e = Enroll face    d = Delete face   ")
    print("  r = Relock         ESC = Quit         ")
    print("  Resolution: {}x{}".format(config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
    print("========================================")
    print("  Open Palm  → Lights    Peace Sign → Door")
    print("  Pointing Up→ AC        Thumb Up   → Window")
    print("  Confirm: Thumb UP=ON  Thumb DOWN=OFF")
    print("========================================\n")

    stop_event = threading.Event()

    t_face = threading.Thread(
        target=face_thread,
        args=(face, buf, state, stop_event),
        daemon=True, name="FaceThread")

    t_gesture = threading.Thread(
        target=gesture_thread,
        args=(gesture, buf, state, mqtt, stop_event),
        daemon=True, name="GestureThread")

    t_face.start()
    t_gesture.start()
    print("[MAIN] Threads started\n")

    fps      = 0.0
    fps_prev = time.time()

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        buf.write_raw(raw)

        # Best available annotated frame
        display = buf.read_gesture()
        if display is None:
            display = buf.read_face()
        if display is None:
            display = raw.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key not in (255, -1):
            state.set_key(key)

        # All overlays on main thread only (safe on Linux)
        face.draw_status_bar(display)
        face.draw_debug(display)

        # Activation feedback banner
        msg, color = state.get_feedback()
        if msg:
            H = display.shape[0]
            cv2.rectangle(display,
                          (0, H//2-35), (display.shape[1], H//2+35),
                          (20,20,20), -1)
            cv2.putText(display, msg, (20, H//2+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        # FPS
        now      = time.time()
        fps      = 0.9*fps + 0.1*(1.0/(now-fps_prev+1e-6))
        fps_prev = now
        gesture.draw_fps(display, fps)

        # MQTT indicator
        ok = mqtt.is_connected()
        cv2.putText(display, "MQTT OK" if ok else "MQTT OFF",
                    (display.shape[1]-120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0,200,0) if ok else (0,0,200), 2)

        cv2.imshow(WIN, display)

    stop_event.set()
    t_face.join(timeout=2)
    t_gesture.join(timeout=2)
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
