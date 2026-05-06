python3 -c "
import cv2, time
for w,h in [(320,240),(424,240),(640,240),(640,360),(640,480)]:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = cap.read()
    actual = f'{frame.shape[1]}x{frame.shape[0]}' if ret else 'fail'
    # FPS test
    t = time.time()
    for _ in range(20): cap.read()
    fps = 20/(time.time()-t)
    cap.release()
    print(f'Request {w}x{h} -> actual {actual} -> {fps:.1f} FPS')
"
