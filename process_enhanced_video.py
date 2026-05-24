import os
import cv2
import time
import torch
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO

# --- PyTorch 2.6+ Compatibility Patch ---
import torch.serialization
_original_torch_load = torch.load
def _torch_load_patch(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_patch
# ----------------------------------------

# Global Queues
read_queue = Queue(maxsize=30)
write_queue = Queue(maxsize=30)

# Pre-calculate Gamma LUT (Gamma 1.2)
GAMMA = 1.2
invGamma = 1.0 / GAMMA
GAMMA_LUT = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

def apply_preprocessing(frame, save_intermediate=False):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    clahe_img = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)
    
    if save_intermediate:
        if not os.path.exists("pipeline"): os.makedirs("pipeline")
        cv2.imwrite("pipeline/1_clahe.jpg", clahe_img)
    
    gamma_img = cv2.LUT(clahe_img, GAMMA_LUT)
    if save_intermediate:
        cv2.imwrite("pipeline/2_gamma.jpg", gamma_img)
        
    return gamma_img

def reader(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        read_queue.put(frame)
    read_queue.put(None)
    cap.release()
    print("Reader: Finished.")

def worker(model_path):
    # Detection device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Worker: Using device: {device}")

    engine_path = model_path.replace(".pt", ".engine")
    
    # Try to export if engine doesn't exist and GPU is available
    if not os.path.exists(engine_path) and device == 0:
        print(f"Worker: Exporting {model_path} to TensorRT (FP16)...")
        try:
            base_model = YOLO(model_path)
            base_model.export(format="engine", device=device, half=True, imgsz=320)
        except Exception as e:
            print(f"Worker: Export failed: {e}. Falling back to original model.")
    
    # Load the best available model
    final_model_path = engine_path if os.path.exists(engine_path) else model_path
    print(f"Worker: Loading {final_model_path}...")
    model = YOLO(final_model_path, task="detect")
    
    frame_count = 0
    while True:
        frame = read_queue.get()
        if frame is None:
            break
        
        start_time = time.time()
        is_first = (frame_count == 0)
        proc_frame = apply_preprocessing(frame, save_intermediate=is_first)
        
        # Inference
        results = model.predict(proc_frame, imgsz=320, verbose=False, device=device)
        res_frame = results[0].plot()
        
        if is_first:
            cv2.imwrite("pipeline/3_yolo_result.jpg", res_frame)
        
        # Performance info
        fps = 1 / (time.time() - start_time)
        engine_type = "TensorRT GPU" if ".engine" in final_model_path else "NCNN CPU"
        cv2.putText(res_frame, f"FPS: {fps:.1f} ({engine_type})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(res_frame, "JETSON ORIN NANO OPTIMIZED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        write_queue.put(res_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Worker: Processed {frame_count} frames...")
            
    write_queue.put(None)
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
    out_video = "assets/videos/tensorrt_output_roomfire41.mp4"
    model_pt = "Deployment/Models/nano.pt"
    
    if not os.path.exists(in_video):
        print(f"Error: {in_video} not found.")
        return

    cap = cv2.VideoCapture(in_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print("Starting Pipeline...")
    t_reader = threading.Thread(target=reader, args=(in_video,))
    t_worker = threading.Thread(target=worker, args=(model_pt,))
    t_writer = threading.Thread(target=writer, args=(out_video, fps_in, width, height))

    t_reader.start()
    t_worker.start()
    t_writer.start()

    t_reader.join()
    t_worker.join()
    t_writer.join()

if __name__ == "__main__":
    run()
