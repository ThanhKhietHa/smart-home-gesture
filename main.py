"""
main.py — Smart Home Face + Gesture Control
============================================
Architecture: 3 threads, NO inference mutex

WHY NO MUTEX:
  Previous versions used threading.Lock() around MediaPipe calls.
  This caused face thread to starve gesture thread — face re-acquires
  the lock immediately after releasing, gesture thread never schedules.
  Result: gesture ran at ~2 FPS, display froze when face disappeared.

  MediaPipe XNNPACK backend handles concurrent calls safely on its own.
  Removing the mutex lets both threads run at their natural rate.

FRAME DEDUPLICATION:
  Face thread tracks the last frame pointer it processed. If buf.raw
  hasn't changed (same object), it sleeps and tries again. This prevents
  re-running inference on the same camera frame 3-4 times per second.

THREAD SCHEDULING:
  LOCKED:   face thread runs every frame, gesture thread sleeps 20ms
  UNLOCKED: face thread runs presence_only every PRESENCE_EVERY frames
            (not every single frame), gesture runs every frame
  Both threads yield with time.sleep after each inference call so the
  OS scheduler can switch threads naturally.
"""

import cv2
import time
import threading
import numpy as np
import config
from face_auth       import FaceAuth
from gesture_control import GestureControl
from mqtt_handler    import MQTTHandler

# How often face presence check runs when unlocked (every N face-thread ticks)
# Higher = more CPU for gesture, lower = faster relock detection
# At 10 FPS camera: 5 = check every 0.5s, fine for 2s grace period
PRESENCE_EVERY = 5


# =====================================================================
# THREAD-SAFE FRAME BUFFER  — uses frame_id to detect new frames
# =====================================================================
class FrameBuffer:
    def __init__(self):
        self._lock        = threading.Lock()
        self._raw         = None
        self._raw_id      = 0        # increments every write_raw
        self._face_out    = None
        self._gesture_out = None

    def write_raw(self, frame):
        with self._lock:
            self._raw    = frame     # no copy — main thread owns it until next write
            self._raw_id += 1

    def read_raw(self):
        """Returns (frame_copy, frame_id). frame_copy is None if no frame yet."""
        with self._lock:
            if self._raw is None:
                return None, 0
            return self._raw.copy(), self._raw_id

    def write_face(self, frame):
        with self._lock:
            self._face_out = frame

    def read_face(self):
        with self._lock:
            return self._face_out.copy() if self._face_out is not None else None

    def write_gesture(self, frame):
        with self._lock:
            self._gesture_out = frame

    def read_gesture(self):
        with self._lock:
            return self._gesture_out.copy() if self._gesture_out is not None else None

    def clear_gesture(self):
        """Called when relocked so stale gesture overlay is not shown."""
        with self._lock:
            self._gesture_out = None


# =====================================================================
# SHARED STATE
# =====================================================================
class SharedState:
    def __init__(self):
        self._lock          = threading.Lock()
        self._unlocked      = False
        self._name          = ""
        self._key           = -1
        self._feedback_msg  = ""
        self._feedback_col  = (0, 255, 0)
        self._feedback_until = 0.0

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
            k = self._key; self._key = -1; return k

    def set_feedback(self, msg, color=(0, 255, 0), duration=1.5):
        with self._lock:
            self._feedback_msg   = msg
            self._feedback_col   = color
            self._feedback_until = time.time() + duration

    def get_feedback(self):
        with self._lock:
            if time.time() < self._feedback_until:
                return self._feedback_msg, self._feedback_col
            return None, None


# =====================================================================
# FACE THREAD
# =====================================================================
def face_thread(face, buf, state, stop_event):
    """
    Runs face recognition / presence check.
    Uses frame_id deduplication — skips if camera hasn't produced a new frame.
    No mutex — MediaPipe handles concurrent access internally.

    LOCKED:   full process_frame every new camera frame (~31ms per call)
    UNLOCKED: process_presence_only every PRESENCE_EVERY ticks (~8ms per call)
              full process_frame every 90 ticks to re-verify identity
    """
    last_id   = 0
    tick      = 0
    was_unlocked = False

    while not stop_event.is_set():
        raw, fid = buf.read_raw()
        if raw is None or fid == last_id:
            time.sleep(0.005)
            continue
        last_id = fid
        tick   += 1

        key      = state.get_key()
        unlocked = state.is_unlocked()

        # Detect relock transition → clear stale gesture overlay
        if was_unlocked and not unlocked:
            buf.clear_gesture()
        was_unlocked = unlocked

        # Key handling always runs first (state transitions before inference)
        face.handle_key(key)

        if unlocked:
            # Every 90 ticks: full re-recognition to verify user still there
            if tick % 90 == 0:
                frame = face.process_frame(raw)
            # Every PRESENCE_EVERY ticks: lightweight presence check
            elif tick % PRESENCE_EVERY == 0:
                frame = face.process_presence_only(raw)
            else:
                # Most ticks: just write raw so gesture thread has fresh base
                buf.write_face(raw)
                state.set_auth(face.is_unlocked(), face.unlocked_name())
                continue
        else:
            # Locked: full recognition every frame
            frame = face.process_frame(raw)

        state.set_auth(face.is_unlocked(), face.unlocked_name())
        buf.write_face(frame)
        # Small yield so gesture thread gets CPU time
        time.sleep(0.002)


# =====================================================================
# GESTURE THREAD
# =====================================================================
def gesture_thread(gesture, buf, state, mqtt, stop_event):
    """
    Runs hand gesture detection when unlocked.
    No mutex — runs concurrently with face thread.
    Uses frame_id on face buffer to avoid reprocessing same frame.
    """
    last_face_frame = None

    while not stop_event.is_set():
        if not state.is_unlocked():
            # Clear gesture state so re-entry starts fresh
            gesture.reset()
            time.sleep(0.020)
            last_face_frame = None
            continue

        base = buf.read_face()
        if base is None:
            base_raw, _ = buf.read_raw()
            base = base_raw
        if base is None:
            time.sleep(0.008)
            continue

        # Skip if base frame looks identical to last processed
        # (cheap pointer/shape check — avoids duplicate inference)
        frame_sig = (base.shape, base[0, 0, 0] if base.size > 0 else 0)
        if frame_sig == last_face_frame:
            time.sleep(0.003)
            continue
        last_face_frame = frame_sig

        frame, feedback = gesture.process_frame(base, mqtt, True)
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

    # Camera init — MJPG mode MUST be set before resolution/FPS
    # Default OpenCV picks YUYV which saturates USB 2.0 at ~10 FPS.
    # MJPEG compresses in-camera; USB carries ~3x less data = 30 FPS possible.
    # Critical order: open -> FOURCC -> resolution -> FPS -> buffersize
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
        return

    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    cap.set(cv2.CAP_PROP_FOURCC,       fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Verify MJPG actually applied
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str    = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    actual_fps    = cap.get(cv2.CAP_PROP_FPS)
    actual_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mjpg_ok       = fourcc_str.strip() == "MJPG"
    fps_ok        = actual_fps >= 25

    print(f"[CAM] Format: {fourcc_str}  ({'OK compressed' if mjpg_ok else 'WARN not MJPG, bandwidth-limited'})" )
    print(f"[CAM] FPS   : {actual_fps:.0f}  ({'OK' if fps_ok else 'WARN still slow'})")
    print(f"[CAM] Size  : {actual_w}x{actual_h}")
    if not mjpg_ok:
        print("[CAM] MJPG rejected — run: v4l2-ctl --list-formats-ext")

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
    print("========================================")
    print("  Open Palm  → Lights  | Thumb Up=ON  Thumb Down=OFF")
    print("  Peace Sign → Door    | Thumb Up=Toggle")
    print("  Pointing Up→ AC      | Thumb Up=ON  Thumb Down=OFF")
    print("  Thumb Up   → Window  | Thumb Up=Up  Thumb Down=Down")
    print("  (hold entry gesture 1.5s, Open Palm always cancels)")
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

    fps = 0.0
    fps_prev = time.time()

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        buf.write_raw(raw)

        # Best available annotated frame — gesture > face > raw
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

        # All drawing on main thread (safe on Linux)
        face.draw_status_bar(display)
        face.draw_debug(display)

        msg, color = state.get_feedback()
        if msg:
            H = display.shape[0]
            cv2.rectangle(display,
                          (0, H // 2 - 35), (display.shape[1], H // 2 + 35),
                          (20, 20, 20), -1)
            cv2.putText(display, msg, (20, H // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        now  = time.time()
        fps  = 0.9 * fps + 0.1 / (now - fps_prev + 1e-6)
        fps_prev = now
        gesture.draw_fps(display, fps)

        ok = mqtt.is_connected()
        cv2.putText(display, "MQTT OK" if ok else "MQTT OFF",
                    (display.shape[1] - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 200, 0) if ok else (0, 0, 200), 2)

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
