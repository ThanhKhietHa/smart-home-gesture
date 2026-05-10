python3 - << 'EOF'
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import mediapipe as mp, numpy as np, time, os
# python3 test.py
model = "models/hand_landmarker.task"   # adjust path if needed
for name, delegate in [("CPU", BaseOptions.Delegate.CPU), ("GPU", BaseOptions.Delegate.GPU)]:
    try:
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model, delegate=delegate),
            running_mode=vision.RunningMode.IMAGE, num_hands=1)
        lm = vision.HandLandmarker.create_from_options(opts)
        blank = np.zeros((360,640,3), dtype=np.uint8)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank)
        t0 = time.time()
        for _ in range(10): lm.detect(img)
        print(f"{name}: OK  avg={((time.time()-t0)/10*1000):.1f}ms")
    except Exception as e:
        print(f"{name}: FAILED — {e}")
EOF
