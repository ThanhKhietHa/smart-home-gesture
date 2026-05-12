"""
main.py — Smart Home Face + Gesture Control
============================================
Architecture: 3 threads, NO inference mutex

  Thread 1 (main)    — camera read + display only, never blocks
  Thread 2 (face)    — face recognition / presence check
  Thread 3 (gesture) — hand gesture detection, only runs when unlocked

Key design decisions:
  - GStreamer pipeline tries nvjpegdec (HW) → jpegdec (CPU) → V4L2 fallback
  - FrameBuffer uses frame_id counter for reliable deduplication
    (id() on .copy() is always different — wrong approach)
  - face_thread deduplicates by frame_id so MediaPipe never runs twice
    on the same camera frame → prevents thermal throttle
  - handle_key() called before process_frame() — new face_auth API
  - process_frame() takes no key param — handle_key() is separate
"""

import cv2
import time
import threading
import config
from face_auth       import FaceAuth
from gesture_control import GestureControl
from mqtt_handler    import MQTTHandler


# =====================================================================
# GSTREAMER CAMERA — hardware JPEG decode when available
# =====================================================================
def _open_camera(cfg):
    pipeline_hw = (
        f"v4l2src device=/dev/video{cfg.CAMERA_INDEX} io-mode=2 ! "
        f"image/jpeg,width={cfg.CAMERA_WIDTH},height={cfg.CAMERA_HEIGHT},"
        f"framerate={cfg.CAMERA_FPS}/1 ! "
        f"nvjpegdec ! videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )
    pipeline_cpu = (
        f"v4l2src device=/dev/video{cfg.CAMERA_INDEX} io-mode=2 ! "
        f"image/jpeg,width={cfg.CAMERA_WIDTH},height={cfg.CAMERA_HEIGHT},"
        f"framerate={cfg.CAMERA_FPS}/1 ! "
        f"jpegdec ! videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline_hw, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[CAM] Hardware JPEG decode (nvjpegdec)")
        return cap
    cap = cv2.VideoCapture(pipeline_cpu, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[CAM] CPU JPEG decode (jpegdec)")
        return cap
    print("[CAM] GStreamer failed — falling back to V4L2")
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_V4L2)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC,      fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          cfg.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


# =====================================================================
# THREAD-SAFE FRAME BUFFER — frame_id for reliable deduplication
# =====================================================================
class FrameBuffer:
    """
    frame_id increments on every write_raw. Background threads compare
    against last_seen_id to skip frames the camera hasn't updated yet.
    Using id() on a .copy() doesn't work — every copy is a new object.
    """
    def __init__(self):
        self._lock        = threading.Lock()
        self._raw         = None
        self._raw_id      = 0
        self._face_out    = None
        self._gesture_out = None

    def write_raw(self, frame):
        with self._lock:
            self._raw    = frame      # no copy — main thread owns until next write
            self._raw_id += 1

    def read_raw(self):
        """Returns (frame_copy_or_None, frame_id)."""
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
        with self._lock:
            self._gesture_out = None


# =====================================================================
# SHARED STATE
# =====================================================================
class SharedState:
    def __init__(self):
        self._lock           = threading.Lock()
        self._unlocked       = False
        self._name           = ""
        self._key            = -1
        self._feedback_msg   = ""
        self._feedback_col   = (0, 255, 0)
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
    Deduplicates by frame_id — never runs MediaPipe twice on the same frame.
    handle_key() called before process_frame() (new face_auth API).
    process_frame() takes no key param.

    LOCKED:   full recognition every new frame (~30ms)
    UNLOCKED: process_presence_only every PRESENCE_EVERY new frames (~8ms)
              full re-verify every 90 frames
    """
    PRESENCE_EVERY = 5
    last_id        = 0
    tick           = 0
    was_unlocked   = False

    while not stop_event.is_set():
        raw, fid = buf.read_raw()
        if raw is None or fid == last_id:
            time.sleep(0.005)
            continue
        last_id = fid
        tick   += 1

        key      = state.get_key()
        unlocked = state.is_unlocked()

        # Clear stale gesture overlay when transitioning to locked
        if was_unlocked and not unlocked:
            buf.clear_gesture()
        was_unlocked = unlocked

        # handle_key runs state transitions before inference
        face.handle_key(key)

        # Write raw first — display never stalls during inference
        buf.write_face(raw)

        if unlocked:
            if tick % 90 == 0:
                frame = face.process_frame(raw)           # periodic re-verify
            elif tick % PRESENCE_EVERY == 0:
                frame = face.process_presence_only(raw)   # lightweight ~8ms
            else:
                # Skip — raw already written above, gesture thread has fresh base
                state.set_auth(face.is_unlocked(), face.unlocked_name())
                continue
        else:
            frame = face.process_frame(raw)               # full recognition

        state.set_auth(face.is_unlocked(), face.unlocked_name())
        buf.write_face(frame)
        time.sleep(0.002)   # yield CPU to gesture thread


# =====================================================================
# GESTURE THREAD
# =====================================================================
def gesture_thread(gesture, buf, state, mqtt, stop_event):
    """
    Runs only when unlocked. No mutex needed — when unlocked the face
    thread is mostly doing ~8ms presence checks, not 30ms MediaPipe,
    so there's no meaningful contention on the XNNPACK backend.
    """
    while not stop_event.is_set():
        if not state.is_unlocked():
            gesture.reset() if hasattr(gesture, 'reset') else None
            time.sleep(0.020)
            continue

        base = buf.read_face()
        if base is None:
            raw, _ = buf.read_raw()
            base = raw
        if base is None:
            time.sleep(0.008)
            continue

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

    cap = _open_camera(config)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
        return

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
            H = display.shape[0]
            cv2.rectangle(display,
                          (0, H // 2 - 35), (display.shape[1], H // 2 + 35),
                          (20, 20, 20), -1)
            cv2.putText(display, msg, (20, H // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        now      = time.time()
        fps      = 0.9 * fps + 0.1 / (now - fps_prev + 1e-6)
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
        # Flags for gesture thread to show activation feedback on main frame
        self.show_feedback    = False
        self.feedback_msg     = ""
        self.feedback_color   = (0, 255, 0)
        self.feedback_until   = 0.0

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

    def set_feedback(self, msg, color=(0,255,0), duration=1.2):
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
    Always writes raw first (freeze fix), then processes on private copy.

    KEY FIX — skip duplicate frames:
      Camera produces 30fps but this thread can loop much faster.
      Running MediaPipe on the same frame repeatedly burns CPU for zero
      new information → thermal throttle → FPS drops from 15 to 10.
      We track the last raw frame's id() and skip if nothing new arrived.

    LOCKED:   runs MediaPipe every frame (new frames only) for fast recognition.
    UNLOCKED: runs check_presence every frame (new frames only) for grace counter.
    """
    last_raw_id = None

    while not stop_event.is_set():
        raw = buf.read_raw()
        if raw is None:
            time.sleep(0.008)
            continue

        # Skip if camera hasn't produced a new frame yet
        raw_id = id(raw)
        if raw_id == last_raw_id:
            time.sleep(0.004)   # yield CPU, wait for next camera frame
            continue
        last_raw_id = raw_id

        # Write raw immediately — display never stalls during inference
        buf.write_face(raw)

        key   = state.get_key()
        work  = raw.copy()
        frame = face.process_frame(work, key)
        face.handle_key(key)
        state.set_auth(face.is_unlocked(), face.unlocked_name())

        buf.write_face(frame)


# =====================================================================
# GESTURE THREAD
# =====================================================================
def gesture_thread(gesture, buf, state, mqtt, stop_event):
    """
    Smart scheduling:
      LOCKED   → skip gesture entirely (face not verified, no point running)
      UNLOCKED → run gesture every frame at full speed
    This frees ~35ms per frame for face when locked.
    """
    while not stop_event.is_set():
        # When locked — no gesture processing needed at all
        if not state.is_unlocked():
            time.sleep(0.02)
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

    # ── Camera — buffer size FIRST, then resolution ───────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # must be set first
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)

    # ── Camera warmup — discard first 10 frames ───────────────────────
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
    print("========================================\n")

    # ── Start threads ─────────────────────────────────────────────────
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

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key not in (255, -1):
            state.set_key(key)

        # ── Draw UI overlays (main thread only — safe on Linux) ───────
        face.draw_status_bar(display)
        face.draw_debug(display)

        # Activation feedback banner (replaces imshow in gesture thread)
        msg, color = state.get_feedback()
        if msg:
            H = display.shape[0]
            cv2.rectangle(display, (0, H//2-35), (display.shape[1], H//2+35),
                          (20,20,20), -1)
            cv2.putText(display, msg,
                        (20, H//2+12), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, color, 3)

        # FPS counter
        now  = time.time()
        fps  = 0.9 * fps + 0.1 * (1.0 / (now - fps_prev + 1e-6))
        fps_prev = now
        gesture.draw_fps(display, fps)

        # MQTT indicator
        ok  = mqtt.is_connected()
        cv2.putText(display, "MQTT OK" if ok else "MQTT OFF",
                    (display.shape[1]-120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0,200,0) if ok else (0,0,200), 2)

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
