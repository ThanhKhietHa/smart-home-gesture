"""
main.py — Smart Home Face + Gesture Control
3 threads: main (display), face, gesture
GStreamer: nvjpegdec (HW) → jpegdec (CPU) → V4L2+MJPG fallback
"""

import cv2
import time
import threading
import config
from face_auth       import FaceAuth
from gesture_control import GestureControl
from mqtt_handler    import MQTTHandler

PRESENCE_EVERY = 5   # face presence check every N ticks when unlocked


# =====================================================================
# CAMERA — GStreamer with HW decode, fallback to V4L2 MJPG
# =====================================================================
def _open_camera(cfg):
    """
    Try 3 methods in order:
    1. GStreamer + nvjpegdec  (Jetson hardware JPEG decoder, ~2ms)
    2. GStreamer + jpegdec    (CPU JPEG decoder, ~8ms)
    3. V4L2 + MJPG fourcc    (OpenCV built-in, ~15ms)
    """
    w, h, fps, idx = cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT, cfg.CAMERA_FPS, cfg.CAMERA_INDEX

    pipeline_hw = (
        f"v4l2src device=/dev/video{idx} io-mode=2 ! "
        f"image/jpeg,width={w},height={h},framerate={fps}/1 ! "
        f"nvjpegdec ! videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )
    pipeline_cpu = (
        f"v4l2src device=/dev/video{idx} io-mode=2 ! "
        f"image/jpeg,width={w},height={h},framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

    cap = cv2.VideoCapture(pipeline_hw, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print("[CAM] Hardware JPEG decode (nvjpegdec) — best performance")
            return cap
        cap.release()

    cap = cv2.VideoCapture(pipeline_cpu, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print("[CAM] CPU JPEG decode (jpegdec)")
            return cap
        cap.release()

    print("[CAM] GStreamer failed — using V4L2 + MJPG")
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC,      fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Verify format actually applied
    actual = int(cap.get(cv2.CAP_PROP_FOURCC))
    fmt = "".join([chr((actual >> 8*i) & 0xFF) for i in range(4)])
    afps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] Format={fmt} FPS={afps:.0f} Size={int(cap.get(3))}x{int(cap.get(4))}")
    if fmt.strip() != 'MJPG':
        print("[CAM] WARN: MJPG not applied — may be bandwidth-limited")
    return cap


# =====================================================================
# FRAME BUFFER — frame_id counter for deduplication
# =====================================================================
class FrameBuffer:
    """
    frame_id increments every write_raw so threads can detect new frames
    without relying on id() which changes on every .copy().
    """
    def __init__(self):
        self._lock        = threading.Lock()
        self._raw         = None
        self._raw_id      = 0
        self._face_out    = None
        self._gesture_out = None

    def write_raw(self, frame):
        with self._lock:
            self._raw    = frame   # no copy — main thread owns until next write
            self._raw_id += 1

    def read_raw(self):
        """Returns (copy, frame_id). copy is None if no frame yet."""
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
    Deduplicates by frame_id — MediaPipe never runs twice on same frame.
    LOCKED:   full recognition every new frame
    UNLOCKED: presence_only every PRESENCE_EVERY frames
              full re-verify every 90 frames
    """
    last_id      = 0
    tick         = 0
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

        # Relock transition — clear stale gesture overlay
        if was_unlocked and not unlocked:
            buf.clear_gesture()
        was_unlocked = unlocked

        # Key handling before inference
        face.handle_key(key)

        # Write raw immediately so display never stalls
        buf.write_face(raw)

        if unlocked:
            if tick % 90 == 0:
                frame = face.process_frame(raw)
            elif tick % PRESENCE_EVERY == 0:
                frame = face.process_presence_only(raw)
            else:
                state.set_auth(face.is_unlocked(), face.unlocked_name())
                continue
        else:
            frame = face.process_frame(raw)

        state.set_auth(face.is_unlocked(), face.unlocked_name())
        buf.write_face(frame)
        time.sleep(0.002)  # yield CPU to gesture thread


# =====================================================================
# GESTURE THREAD
# =====================================================================
def gesture_thread(gesture, buf, state, mqtt, stop_event):
    """
    Only runs when unlocked. No mutex needed.
    """
    while not stop_event.is_set():
        if not state.is_unlocked():
            gesture.reset()
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
        print("[ERROR] Cannot open camera.")
        return

    # Warmup — discard first 15 frames (exposure stabilisation)
    print("[MAIN] Camera warming up...")
    for _ in range(15):
        cap.read()

    WIN = "Smart Home"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, face.mouse_callback)

    print("\n========================================")
    print("  Smart Home — Face + Gesture Control  ")
    print("========================================")
    print("  e = Enroll face    d = Delete face   ")
    print("  r = Relock         ESC = Quit         ")
    print(f"  Resolution: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
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

        # Best available frame: gesture > face > raw
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

        # All drawing on main thread only (Linux X11 requirement)
        face.draw_status_bar(display)
        face.draw_debug(display)

        msg, color = state.get_feedback()
        if msg:
            H = display.shape[0]
            cv2.rectangle(display,
                          (0, H//2 - 35), (display.shape[1], H//2 + 35),
                          (20, 20, 20), -1)
            cv2.putText(display, msg, (20, H//2 + 12),
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
