# main_optimized.py - Parallel Face + Hand Detection
import cv2
import time
import threading
import numpy as np
import config
from face_auth import FaceAuth
from gesture_control import GestureControl
from mqtt_handler import MQTTHandler

class FrameBuffer:
    def __init__(self, maxsize=2):
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._frames = []
        
    def put(self, frame):
        with self._lock:
            self._frames.append(frame)
            if len(self._frames) > self.maxsize:
                self._frames.pop(0)
    
    def get(self):
        with self._lock:
            if self._frames:
                return self._frames[-1].copy()  # Latest frame
            return None
    
    def get_oldest(self):
        with self._lock:
            if self._frames:
                return self._frames.pop(0).copy()
            return None

def main():
    mqtt = MQTTHandler()
    face = FaceAuth()
    gesture = GestureControl()
    
    # Shared buffers for parallel processing
    raw_buffer = FrameBuffer(maxsize=2)
    face_buffer = FrameBuffer(maxsize=1)
    gesture_buffer = FrameBuffer(maxsize=1)
    
    # Shared state
    state = {
        'unlocked': False,
        'running': True,
        'face_frame_count': 0,
        'gesture_frame_count': 0
    }
    state_lock = threading.Lock()
    
    # ===== THREAD 1: Camera Capture (Highest Priority) =====
    def camera_thread():
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        # Warm up
        for _ in range(5):
            cap.read()
        
        while state['running']:
            ret, frame = cap.read()
            if ret:
                raw_buffer.put(frame)
            else:
                time.sleep(0.001)
        cap.release()
    
    # ===== THREAD 2: Face Recognition (Runs in parallel with gesture) =====
    def face_thread():
        frame_skip = 0
        while state['running']:
            frame = raw_buffer.get_oldest()
            if frame is None:
                time.sleep(0.001)
                continue
            
            with state_lock:
                unlocked = state['unlocked']
            
            # Smart scheduling
            if not unlocked:
                # When locked: process every frame (need fast unlock)
                frame_skip = 0
            else:
                # When unlocked: process every 60th frame (saves CPU for gesture)
                frame_skip += 1
                if frame_skip < 60:
                    continue
                frame_skip = 0
            
            try:
                # Process face (takes ~32ms)
                result = face.process_frame(frame, -1)
                face_buffer.put(result)
                
                # Update unlock state
                with state_lock:
                    state['unlocked'] = face.is_unlocked()
                    if state['unlocked']:
                        print(f"[FACE] Unlocked: {face.unlocked_name()}")
            except Exception as e:
                print(f"Face error: {e}")
    
    # ===== THREAD 3: Gesture Control (Runs in parallel with face) =====
    def gesture_thread():
        frame_skip = 0
        while state['running']:
            with state_lock:
                unlocked = state['unlocked']
            
            # Only run gesture when unlocked
            if not unlocked:
                time.sleep(0.02)
                continue
            
            frame = raw_buffer.get_oldest()
            if frame is None:
                time.sleep(0.001)
                continue
            
            # Run gesture every 2nd frame for speed
            frame_skip += 1
            if frame_skip < 2:
                continue
            frame_skip = 0
            
            try:
                # Process gesture in parallel with face (takes ~45ms)
                result, feedback = gesture.process_frame(frame, mqtt, unlocked)
                gesture_buffer.put(result)
                
                # Show feedback on main frame
                if feedback:
                    # We'll handle feedback in display thread
                    pass
            except Exception as e:
                print(f"Gesture error: {e}")
    
    # ===== THREAD 4: Display (Always shows latest frame) =====
    def display_thread():
        fps = 0
        fps_counter = 0
        fps_time = time.time()
        feedback_msg = None
        feedback_color = (0, 255, 0)
        feedback_end = 0
        
        cv2.namedWindow("Smart Home", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Smart Home", face.mouse_callback)
        
        while state['running']:
            # Get the most processed frame available
            display = gesture_buffer.get()
            if display is None:
                display = face_buffer.get()
            if display is None:
                display = raw_buffer.get()
            if display is None:
                time.sleep(0.001)
                continue
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Draw minimal UI (optimized for speed)
            with state_lock:
                unlocked = state['unlocked']
            
            # Status bar (simplified)
            h, w = display.shape[:2]
            if unlocked:
                cv2.rectangle(display, (0, 0), (w, 45), (0, 80, 0), -1)
                cv2.putText(display, f"UNLOCKED", (10, 32), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(display, (0, 0), (w, 45), (0, 0, 80), -1)
                cv2.putText(display, f"LOCKED", (10, 32), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
            
            # FPS counter
            cv2.putText(display, f"{fps} FPS", (w - 70, 32), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Debug info (optional - remove for more speed)
            cv2.putText(display, f"F:{face._last_se:.2f} G:{face._last_ie:.3f}", 
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Show frame
            cv2.imshow("Smart Home", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                state['running'] = False
                break
            elif key == ord('e'):
                face.handle_key(ord('e'))
            elif key == ord('d'):
                face.handle_key(ord('d'))
            elif key == ord('r'):
                face.handle_key(ord('r'))
    
    # Start all threads
    threads = [
        threading.Thread(target=camera_thread, name="Camera", daemon=True),
        threading.Thread(target=face_thread, name="Face", daemon=True),
        threading.Thread(target=gesture_thread, name="Gesture", daemon=True),
        threading.Thread(target=display_thread, name="Display", daemon=True),
    ]
    
    print("\n" + "="*50)
    print("  SMART HOME - PARALLEL MODE")
    print("  Face: 32ms | Gesture: 45ms | Running in parallel")
    print("="*50)
    print("  Expected FPS: 18-20")
    print("  e=Enroll d=Delete r=Relock ESC=Quit")
    print("="*50 + "\n")
    
    for t in threads:
        t.start()
    
    # Keep main thread alive
    try:
        while state['running']:
            time.sleep(0.1)
    except KeyboardInterrupt:
        state['running'] = False
    
    cv2.destroyAllWindows()
    mqtt.stop()
    print("\nProgram ended.")

if __name__ == '__main__':
    main()
