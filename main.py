"""
main.py — Optimized for Jetson Orin Nano
"""

import cv2
import time
import threading
import config
from face_auth import FaceAuth
from gesture_control import GestureControl
from mqtt_handler import MQTTHandler

class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._raw = None
        self._face_out = None
        self._gesture_out = None

    def write_raw(self, frame):
        with self._lock:
            self._raw = frame.copy()

    def read_raw(self):
        with self._lock:
            return self._raw.copy() if self._raw is not None else None

    def write_face(self, frame):
        with self._lock:
            self._face_out = frame.copy()

    def read_face(self):
        with self._lock:
            return self._face_out.copy() if self._face_out is not None else None

    def write_gesture(self, frame):
        with self._lock:
            self._gesture_out = frame.copy()

    def read_gesture(self):
        with self._lock:
            return self._gesture_out.copy() if self._gesture_out is not None else None

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._unlocked = False
        self._name = ""
        self._key = -1
        self.show_feedback = False
        self.feedback_msg = ""
        self.feedback_color = (0, 255, 0)
        self.feedback_until = 0.0

    def set_auth(self, unlocked, name):
        with self._lock:
            self._unlocked = unlocked
            self._name = name

    def is_unlocked(self):
        with self._lock:
            return self._unlocked

    def set_key(self, key):
        with self._lock:
            self._key = key

    def get_key(self):
        with self._lock:
            k = self._key
            self._key = -1
            return k

    def set_feedback(self, msg, color=(0,255,0), duration=1.0):
        with self._lock:
            self.show_feedback = True
            self.feedback_msg = msg
            self.feedback_color = color
            self.feedback_until = time.time() + duration

    def get_feedback(self):
        with self._lock:
            if self.show_feedback and time.time() < self.feedback_until:
                return self.feedback_msg, self.feedback_color
            self.show_feedback = False
            return None, None

def face_thread(face, buf, state, stop_event):
    frame_counter = 0
    frame_delay = 0.01  # 10ms delay between frames
    
    while not stop_event.is_set():
        raw = buf.read_raw()
        if raw is None:
            time.sleep(0.005)
            continue

        frame_counter += 1
        key = state.get_key()
        unlocked = state.is_unlocked()

        # Run less frequently when unlocked
        if unlocked and frame_counter % 60 != 0 and key == -1:
            time.sleep(frame_delay)
            continue

        start = time.time()
        frame = face.process_frame(raw, key)
        face.handle_key(key)
        state.set_auth(face.is_unlocked(), face.unlocked_name())
        buf.write_face(frame)
        
        elapsed = time.time() - start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

def gesture_thread(gesture, buf, state, mqtt, stop_event):
    frame_counter = 0
    frame_delay = 0.01
    
    while not stop_event.is_set():
        if not state.is_unlocked():
            time.sleep(0.03)
            continue

        frame_counter += 1
        if frame_counter % config.GESTURE_EVERY_N_FRAMES != 0:
            time.sleep(frame_delay)
            continue

        base = buf.read_face()
        if base is None:
            base = buf.read_raw()
        if base is None:
            time.sleep(0.005)
            continue

        start = time.time()
        frame, feedback = gesture.process_frame(base, mqtt, state.is_unlocked())
        if feedback:
            state.set_feedback(feedback[0], feedback[1])
        buf.write_gesture(frame)
        
        elapsed = time.time() - start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

def main():
    mqtt = MQTTHandler()
    face = FaceAuth()
    gesture = GestureControl()
    buf = FrameBuffer()
    state = SharedState()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    print("[MAIN] Camera:", config.CAMERA_WIDTH, "x", config.CAMERA_HEIGHT)
    for _ in range(5):
        cap.read()

    WIN = "Smart Home"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, face.mouse_callback)

    print("\n" + "="*40)
    print("  Smart Home Control - OPTIMIZED")
    print("="*40)
    print("  e=Enroll  d=Delete  r=Relock  ESC=Quit")
    print("="*40 + "\n")

    stop_event = threading.Event()
    t_face = threading.Thread(target=face_thread, args=(face, buf, state, stop_event), daemon=True)
    t_gesture = threading.Thread(target=gesture_thread, args=(gesture, buf, state, mqtt, stop_event), daemon=True)
    t_face.start()
    t_gesture.start()

    fps = 0.0
    fps_prev = time.time()

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        buf.write_raw(raw)

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

        face.draw_status_bar(display)
        face.draw_debug(display)

        msg, color = state.get_feedback()
        if msg:
            h = display.shape[0]
            cv2.rectangle(display, (0, h//2-30), (display.shape[1], h//2+30), (20,20,20), -1)
            cv2.putText(display, msg, (20, h//2+8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - fps_prev + 1e-6))
        fps_prev = now
        gesture.draw_fps(display, fps)

        ok = mqtt.is_connected()
        cv2.putText(display, "MQTT", (display.shape[1]-55, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0) if ok else (0,0,200), 2)

        cv2.imshow(WIN, display)

    stop_event.set()
    t_face.join(timeout=1)
    t_gesture.join(timeout=1)
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
        input("Press Enter...")
