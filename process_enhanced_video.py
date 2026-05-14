import os
# Force CPU mode for consistency
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import time
import torch
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO

# 1. PyTorch 2.6+ Patch for legacy YOLO loading
_original_torch_load = torch.load
def _torch_load_patch(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_patch

# Global Queues
read_queue = Queue(maxsize=30)
write_queue = Queue(maxsize=30)

# Pre-calculate Gamma LUT (Gamma 1.2)
GAMMA = 1.2
invGamma = 1.0 / GAMMA
GAMMA_LUT = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

def apply_preprocessing(frame):
    # CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    processed = cv2.merge((cl, a, b))
    processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
    
    # Apply pre-calculated Gamma LUT
    processed = cv2.LUT(processed, GAMMA_LUT)
    return processed

def reader(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        read_queue.put(frame)
    read_queue.put(None) # Signal end
    cap.release()
    print("Reader: Finished.")

def worker(model_path):
    # Load model (Prefer NCNN if exists)
    ncnn_path = model_path.replace(".pt", "_ncnn_model")
    if not os.path.exists(ncnn_path):
        print(f"Worker: Exporting {model_path} to NCNN INT8...")
        base_model = YOLO(model_path)
        base_model.export(format="ncnn", int8=True, imgsz=320)
    
    model = YOLO(ncnn_path, task="detect")
    print(f"Worker: Using NCNN model from {ncnn_path}")
    
    frame_count = 0
    while True:
        frame = read_queue.get()
        if frame is None:
            break
        
        start_time = time.time()
        
        # 1. Preprocess
        proc_frame = apply_preprocessing(frame)
        
        # 2. Predict (imgsz=320 is critical for NCNN speed)
        results = model.predict(proc_frame, imgsz=320, verbose=False)
        
        # 3. Plot
        res_frame = results[0].plot()
        
        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(res_frame, f"FPS: {fps:.1f} (NCNN INT8)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(res_frame, "PRODUCER-CONSUMER PIPELINE", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        write_queue.put(res_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Worker: Processed {frame_count} frames...")
            
    write_queue.put(None) # Signal end
    print(f"Worker: Finished. Total {frame_count} frames.")

def writer(out_path, fps_in, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
    
    while True:
        frame = write_queue.get()
        if frame is None:
            break
        out.write(frame)
    
    out.release()
    print(f"Writer: Finished. Saved to {out_path}")

def run():
    in_video = "assets/videos/roomfire41.mp4"
    out_video = "assets/videos/output_roomfire41.mp4"
    model_pt = "Deployment/Models/nano.pt"
    
    if not os.path.exists(in_video):
        print(f"Error: {in_video} not found.")
        return

    # Get video info for writer
    cap = cv2.VideoCapture(in_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print("Starting Optimized Pipeline (NCNN INT8 + Multi-threading)...")
    
    t_reader = threading.Thread(target=reader, args=(in_video,))
    t_worker = threading.Thread(target=worker, args=(model_pt,))
    t_writer = threading.Thread(target=writer, args=(out_video, fps_in, width, height))

    t_reader.start()
    t_worker.start()
    t_writer.start()

    t_reader.join()
    t_worker.join()
    t_writer.join()
    
    print("Optimization Test Complete.")

if __name__ == "__main__":
    run()
