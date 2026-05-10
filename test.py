#!/usr/bin/env python3
"""Test camera with all fixes: disable auto-exposure, force MJPG, threaded"""
#v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG
#v4l2-ctl -d /dev/video0 --set-parm=30
#v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1
#v4l2-ctl -d /dev/video0 --set-ctrl=exposure_time_absolute=100
#python3 test.py
import cv2
import time
import threading

class FastCamera:
    def __init__(self, device=0, width=640, height=480, fps=30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.frame = None
        self.running = True
        
    def start(self):
        # Pipeline GStreamer với hardware decode
        pipeline = (
            f"v4l2src device=/dev/video{self.device} io-mode=2 ! "
            f"image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            f"jpegdec ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=1"
        )
        
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("[ERROR] Cannot open camera")
            return False
        
        # Tắt auto exposure qua V4L2 (cách khác)
        import subprocess
        subprocess.run([
            "v4l2-ctl", "-d", f"/dev/video{self.device}",
            "--set-ctrl=auto_exposure=1"
        ], capture_output=True)
        
        # Start capture thread
        self.thread = threading.Thread(target=self._update)
        self.thread.start()
        return True
    
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.001)
    
    def read(self):
        return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()

# Test
print("[TEST] Starting camera with fixes...")
cam = FastCamera()
cam.start()

time.sleep(1)

start = time.time()
count = 0
while time.time() - start < 5:
    frame = cam.read()
    if frame is not None:
        count += 1
    time.sleep(0.001)

fps = count / 5
print(f"[RESULT] FPS: {fps:.1f}")

cam.stop()
