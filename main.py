"""
main.py — Smart Home Face + Gesture Control
============================================
Architecture: 3 threads + inference mutex

  Thread 1 (main)    — camera read + display only, never blocks
  Thread 2 (face)    — face recognition / presence check
  Thread 3 (gesture) — hand gesture detection

KEY FIX — Inference Mutex:
  MediaPipe models share a single-threaded XNNPACK backend on Jetson.
  When face thread runs full recognition (~31ms) at the same time as
  gesture thread runs hand detection (~45ms), they serialise internally
  BUT the combined blocking time causes a visible freeze (~76ms = 1.3 FPS).

  Solution: a threading.Lock() (infer_lock) shared between both threads.
  Only one MediaPipe inference runs at a time.
  Face thread acquires lock → runs model → releases.
  Gesture thread acquires lock → runs model → releases.
  This eliminates the freeze when face re-appears (frame % 90 == 0).

KEY FIX — Grace Period:
  Previously: face_thread wrote raw frame and skipped entirely when
  unlocked + frame % 90 != 0 → grace_count never incremented → system
  stayed unlocked forever if user walked away.

  Fix: face_thread calls face.process_presence_only(raw) on skip frames.
  This runs face detector (~8ms), updates tracker, increments grace_count.
  Relock happens correctly after GRACE_FRAMES (20 frames ≈ 2s) with no face.

FPS target: 15-20 FPS unlocked (single model per frame), 10-12 FPS locked.
"""

import cv2
import time
import threading
import numpy as np
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


# =====================================================================
# SHARED STATE
# =====================================================================
class SharedState:
    def __init__(self):
        self._lock     = threading.Lock()
        self._unlocked = False
        self._name     = ""
        self._key      = -1
        self.show_feedback  = False
        self.feedback_msg   = ""
        self.feedback_color = (0, 255, 0)
        self.feedback_until = 0.0

    def set_auth(self, unlocked, name):
        with self._lock:
            self._unlocked = unlocked
            self._name     = name

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

    def set_feedback(self, msg, color=(0, 255, 0), duration=1.5):
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
def face_thread(face, buf, state, infer_lock, stop_event):
    """
    LOCKED:
      Runs full face recognition every FACE_PROCESS_EVERY_N_FRAMES_LOCKED frames.
      Acquires infer_lock before calling MediaPipe.

    UNLOCKED + skip frame (frame % 90 != 0):
      Calls face.process_presence_only(raw) — lightweight ~8ms presence check.
      Acquires infer_lock. Updates tracker + grace_count every frame.
      FIX: this is what prevents the system staying unlocked forever.

    UNLOCKED + recognition frame (frame % 90 == 0):
      Runs full process_frame to check if the recognised person is still there.
      Acquires infer_lock.
    """
    frame_n = 0
    while not stop_event.is_set():
        raw = buf.read_raw()
        if raw is None:
            time.sleep(0.008)
            continue

        frame_n += 1
        key      = state.get_key()
        unlocked = state.is_unlocked()

        if unlocked and frame_n % config.FACE_PROCESS_EVERY_N_FRAMES_UNLOCKED != 0 and key == -1:
            # ── Lightweight presence-only path (FIX for grace bug + freeze) ──
            with infer_lock:
                frame = face.process_presence_only(raw)
            state.set_auth(face.is_unlocked(), face.unlocked_name())
            buf.write_face(frame)
            time.sleep(0.004)   # yield to gesture thread between frames
            continue

        # ── Full recognition path ─────────────────────────────────────
        with infer_lock:
            frame = face.process_frame(raw, key)
        face.handle_key(key)
        state.set_auth(face.is_unlocked(), face.unlocked_name())
        buf.write_face(frame)


# =====================================================================
# GESTURE THREAD
# =====================================================================
def gesture_thread(gesture, buf, state, mqtt, infer_lock, stop_event):
    """
    LOCKED: sleeps — face not verified, no gesture processing needed.
            Saves ~45ms per frame for face thread.

    UNLOCKED: runs hand detection every frame.
              Acquires infer_lock — never overlaps with face inference.
              This eliminates the freeze when face re-appears at frame % 90.

    Performance (unlocked):
      face thread:    ~8ms  (presence_only, every frame except 1-in-90)
      gesture thread: ~45ms (hand detection, every frame)
      With mutex:     ~53ms total serialised = ~18 FPS
      Without mutex:  both models try to run simultaneously on same backend
                      = spike to ~76ms = freeze visible to user
    """
    while not stop_event.is_set():
        if not state.is_unlocked():
            time.sleep(0.020)
            continue

        # Use face-annotated frame as base (has auth box drawn on it)
        base = buf.read_face()
        if base is None:
            base = buf.read_raw()
        if base is None:
            time.sleep(0.008)
            continue

        with infer_lock:
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

    # ── Inference mutex — single MediaPipe inference at a time ────────
    infer_lock = threading.Lock()

    # ── Camera ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)

    # Discard first 10 frames (camera warmup)
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
    print("\n  GESTURE FLOW:")
    print("  Open Palm  (hold) → Lights menu  → Thumb Up=ON  / Thumb Down=OFF  / Open Palm=Cancel")
    print("  Peace Sign (hold) → Door menu    → Thumb Up=Toggle              / Open Palm=Cancel")
    print("  Pointing Up(hold) → AC menu      → Thumb Up=ON  / Thumb Down=OFF  / Open Palm=Cancel")
    print("  Thumb Up   (hold) → Window menu  → Thumb Up=Up  / Thumb Down=Down / Open Palm=Cancel")
    print("========================================\n")

    # ── Start background threads ──────────────────────────────────────
    stop_event = threading.Event()

    t_face = threading.Thread(
        target=face_thread,
        args=(face, buf, state, infer_lock, stop_event),
        daemon=True, name="FaceThread")

    t_gesture = threading.Thread(
        target=gesture_thread,
        args=(gesture, buf, state, mqtt, infer_lock, stop_event),
        daemon=True, name="GestureThread")

    t_face.start()
    t_gesture.start()
    print("[MAIN] Threads started\n")

    # ── FPS — exponential moving average ──────────────────────────────
    fps      = 0.0
    fps_prev = time.time()

    # ── Display loop ──────────────────────────────────────────────────
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

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key not in (255, -1):
            state.set_key(key)

        # ── Overlays drawn on main thread only (safe on Linux) ────────
        face.draw_status_bar(display)
        face.draw_debug(display)

        # Activation feedback banner
        msg, color = state.get_feedback()
        if msg:
            H = display.shape[0]
            cv2.rectangle(display,
                          (0, H // 2 - 35), (display.shape[1], H // 2 + 35),
                          (20, 20, 20), -1)
            cv2.putText(display, msg,
                        (20, H // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        # FPS counter
        now      = time.time()
        fps      = 0.9 * fps + 0.1 * (1.0 / (now - fps_prev + 1e-6))
        fps_prev = now
        gesture.draw_fps(display, fps)

        # MQTT status indicator
        ok = mqtt.is_connected()
        cv2.putText(display, "MQTT OK" if ok else "MQTT OFF",
                    (display.shape[1] - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 200, 0) if ok else (0, 0, 200), 2)

        cv2.imshow(WIN, display)

    # ── Cleanup ───────────────────────────────────────────────────────
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
