# Test real FPS with current settings
cd ~/smart-home-gesture
python3 -c "
import cv2
import time
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
start = time.time()
for i in range(100):
    ret, frame = cap.read()
print(f'Camera FPS: {100/(time.time()-start):.1f}')
cap.release()
"
