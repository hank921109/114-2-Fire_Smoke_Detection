import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# --- PyTorch Compatibility Patch ---
import torch.serialization
_original_torch_load = torch.load
def _torch_load_patch(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_patch
# ----------------------------------------

# Setup directories
if not os.path.exists("pipeline"):
    os.makedirs("pipeline")

image_path = "assets/images/smoke_fire_true_positive_nano.jfif"
model_path = "Deployment/Models/nano.pt"

if not os.path.exists(image_path):
    print(f"Error: {image_path} not found.")
    exit(1)

# Pre-calculate Gamma LUT
GAMMA = 1.2
invGamma = 1.0 / GAMMA
GAMMA_LUT = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

frame = cv2.imread(image_path)

# CLAHE
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l)
clahe_img = cv2.merge((cl, a, b))
clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)
cv2.imwrite("pipeline/1_clahe.jpg", clahe_img)

# Gamma
gamma_img = cv2.LUT(clahe_img, GAMMA_LUT)
cv2.imwrite("pipeline/2_gamma.jpg", gamma_img)

# YOLO inference
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(model_path, task="detect")
results = model.predict(gamma_img, imgsz=320, verbose=False, device=device)
res_frame = results[0].plot()
cv2.imwrite("pipeline/3_yolo_result.jpg", res_frame)

print("Pipeline execution completed for single image.")
