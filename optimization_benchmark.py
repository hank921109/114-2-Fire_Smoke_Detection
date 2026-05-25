import os
import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

# --- 全域優化物件 (Hardware-Aware LUT & CLAHE Reuse) ---
GAMMA = 1.2
invGamma = 1.0 / GAMMA
GAMMA_LUT = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
CLAHE_OBJ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def check_roi_presence(frame, threshold=0.005):
    """
    色彩感知感興趣區域 (Color-based ROI Masking)
    快速檢查畫面中是否有疑似火或煙的色彩
    """
    # 縮放以加速檢查 (64x64 足以判斷色彩分布)
    small_frame = cv2.resize(frame, (64, 64))
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # 火災色域 (紅/橙/黃)
    lower_fire1 = np.array([0, 100, 100])
    upper_fire1 = np.array([25, 255, 255])
    lower_fire2 = np.array([160, 100, 100])
    upper_fire2 = np.array([180, 255, 255])
    
    # 煙霧色域 (灰色系，飽和度低，亮度中高)
    lower_smoke = np.array([0, 0, 100])
    upper_smoke = np.array([180, 50, 200])
    
    mask_fire = cv2.bitwise_or(cv2.inRange(hsv, lower_fire1, upper_fire1), 
                               cv2.inRange(hsv, lower_fire2, upper_fire2))
    mask_smoke = cv2.inRange(hsv, lower_smoke, upper_smoke)
    
    combined = cv2.bitwise_or(mask_fire, mask_smoke)
    presence_ratio = np.count_nonzero(combined) / (64 * 64)
    
    return presence_ratio > threshold

def original_preprocessing(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # 每次都建立物件 (耗時)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    clahe_img = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)
    
    # 雖然用了 LUT，但在 Streamlit 等版本中常被重複計算
    invGamma = 1.0 / 1.2
    lut = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(clahe_img, lut)

def optimized_preprocessing(frame):
    # 重用全域 CLAHE 物件
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = CLAHE_OBJ.apply(l)
    clahe_img = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)
    
    # 使用全域預計算 LUT
    return cv2.LUT(clahe_img, GAMMA_LUT)

def benchmark():
    video_path = "assets/videos/roomfire41.mp4"
    model_path = "Deployment/Models/nano.pt"
    if not os.path.exists(video_path):
        print("Video not found.")
        return

    model = YOLO(model_path)
    device = 0 if torch.cuda.is_available() else "cpu"
    
    for mode in ["Original", "Optimized"]:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_time = 0
        inference_count = 0
        
        print(f"\n--- Testing Mode: {mode} ---")
        
        while frame_count < 100: # 測試前 100 幀
            ret, frame = cap.read()
            if not ret: break
            
            start = time.time()
            
            if mode == "Original":
                proc_frame = original_preprocessing(frame)
                results = model.predict(proc_frame, imgsz=320, verbose=False, device=device)
                inference_count += 1
            else:
                # 1. ROI Masking 檢查
                is_interesting = check_roi_presence(frame)
                if is_interesting:
                    # 2. 優化後的 Preprocessing (Reuse objects)
                    proc_frame = optimized_preprocessing(frame)
                    results = model.predict(proc_frame, imgsz=320, verbose=False, device=device)
                    inference_count += 1
                else:
                    # 跳過推論，模擬背景幀處理
                    pass
            
            total_time += (time.time() - start)
            frame_count += 1
        
        avg_fps = frame_count / total_time
        print(f"Results for {mode}:")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Total Frames: {frame_count}")
        print(f"  Inferences Performed: {inference_count}")
        print(f"  Inference Skip Rate: {(1 - inference_count/frame_count)*100:.1f}%")
        
        cap.release()

if __name__ == "__main__":
    benchmark()
