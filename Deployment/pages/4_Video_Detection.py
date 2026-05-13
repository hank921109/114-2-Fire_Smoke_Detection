import os
import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Enhanced Video Detection", layout="wide")

def load_model(model_name="nano"):
    try:
        path = os.path.join("Deployment", "Models", f"{model_name}.pt")
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def apply_preprocessing(frame, enable_clahe, clahe_limit, enable_gamma, gamma_val):
    processed = frame.copy()
    
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if enable_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8,8))
        cl = clahe.apply(l)
        processed = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
    
    # 2. Gamma Correction (np.power based LUT for speed)
    if enable_gamma:
        invGamma = 1.0 / gamma_val
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed = cv2.LUT(processed, table)
        
    return processed

def video_detection():
    st.title("🔥 AI Fire Detection (Enhanced Pipeline)")
    st.write("Using CLAHE + Gamma Correction for better Smoke detection.")

    # Sidebar UI
    st.sidebar.title("Core Settings")
    model_type = st.sidebar.radio("Model:", ("nano", "small"), index=0)
    conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
    
    st.sidebar.divider()
    st.sidebar.title("Preprocessing (Scheme C)")
    
    en_clahe = st.sidebar.checkbox("Enable CLAHE (Contrast Boost)", value=True)
    clahe_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.5)
    
    en_gamma = st.sidebar.checkbox("Enable Gamma Correction", value=True)
    gamma_val = st.sidebar.slider("Gamma Value", 0.5, 3.0, 1.2, 0.1)

    model = load_model(model_type)
    video_path = "assets/videos/roomfire41.mp4"
    
    if not os.path.exists(video_path):
        st.error("Video not found.")
        st.stop()

    cap = cv2.VideoCapture(video_path)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Frame")
        raw_placeholder = st.empty()
    with col2:
        st.subheader("Processed & Detected")
        proc_placeholder = st.empty()

    stop_btn = st.sidebar.button("Stop")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        start_time = time.time()

        # Apply Preprocessing
        enhanced_frame = apply_preprocessing(frame, en_clahe, clahe_limit, en_gamma, gamma_val)

        # Inference on ENHANCED frame
        results = model.predict(enhanced_frame, conf=conf_threshold, imgsz=320, device="cpu", verbose=False)
        annotated_frame = results[0].plot()

        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update UI
        raw_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        proc_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)

    cap.release()

if __name__ == "__main__":
    video_detection()
