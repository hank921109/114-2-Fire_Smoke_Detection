# Fire Detection in Mediterranean Olive Groves (YOLOv8)

針對地中海橄欖園等野外場景，提供早期火災與煙霧的物件偵測（Object Detection）功能。

## 1. 需求 (Requirements)

### 功能
* **核心功能**：提供早期火災與煙霧的物件偵測。
* **模型支援**：同時支援 YOLOv8 Nano 與 Small 兩種權重模型供使用者切換。

### 效能
* **速度**：要求 FPS 達到3 以上。
* **特性**：
    * **Nano 模型**：推論速度較快但精準度稍低。
    * **Small 模型**：速度稍慢但擁有較高的精確度與信心水準。

### 限制與環境
* **環境**：Python 3.10+ (支援 ARM64 架構如 Raspberry Pi)。
* **硬體**：
    * **開發與訓練**：Nvidia RTX 3070 Ti (CUDA)。
    * **部署與推論**：支援 Raspberry Pi 4 (CPU) 或其他邊緣運算裝置。
* **界面**：採用 Streamlit 構建的 Web UI。

### 邊緣運算優化 (Edge AI Optimization)
針對 **Raspberry Pi 4** 等效能有限的裝置，本專案提供以下優化措施：
* **尺寸優化**：建議將 `imgsz` 調降至 **320**，以平衡推論延遲與精確度。
* **模型轉換**：支援轉換為 **NCNN** 格式，顯著提升在 ARM CPU 上的推論效率。

### 界面
* **檔案輸入 (File Input)**：支援從本機上傳圖片（jpg, jpeg, png）。
* **網址輸入 (Stream/URL)**：支援直接貼上圖片 URL 下載並進行推論。
* **內建圖庫**：可隨機抽取內建的 Croatia Fire Dataset (CFD) 進行測試。

### 驗收計畫
* **測試資料**：D-Fire Dataset（超過 21,000 張圖片）與 Croatia Fire Dataset（超過 50 張特定海岸景觀圖）。
* **測試條件**：預設交集聯集比（IOU Threshold）為 0.4，信心門檻（Confidence Threshold）為 0.2（使用者可透過 Slider 動態調整 0.0 ~ 1.0）。
* **期待輸出**：疊加了標註框（Bounding Boxes）的 RGB 影像，以及文字總結（例如："Predicted 2 fires and 1 smoke in 0.15 seconds."），並提供下載預測圖片的功能。

### 如何測試 (Design of Experiment - DOE)
1. 啟動 Streamlit App。
2. 選擇測試模型（Nano 或 Small）。
3. 調變 IOU 與 Confidence Threshold 觀察 False Positive 與 False Negative 變化。
4. 輸入測試圖片（特別針對帶有輕微煙霧的場景）。
5. 比較 Nano 與 Small 模型在同一張圖片上的偵測數量與信心分數。

---

## 2. 分析 (Analysis)

### 系統模組架構 (System Breakdown)
下圖展示了系統的模組化拆解，並對應 DFD 中的資料處理流程：

```mermaid
graph LR
    System[Fire Detection System]
    
    System --> UI[Frontend: Streamlit]
    UI --> UI1["st.file_uploader() / requests.get()"]
    UI --> UI2["Image.open() / PIL Preprocessing"]
    
    System --> Core[Inference: YOLOv8 Engine]
    Core --> Core1["model.predict(imgsz=320)"]
    
    System --> Post[Post-processing: OpenCV / Numpy]
    Post --> Post1["results[0].plot() (Bounding Boxes)"]
    Post --> Post2["cv2.putText() (FPS Overlay)"]
    Post --> Post3["res_image[..., ::-1] (BGR to RGB)"]
```

---

## 3. 設計 (Design)

### Data Flow Diagram (資料流圖)
```mermaid
graph TD
    A[使用者上傳圖片 / URL] -- "Image File / Bytes" --> B(Streamlit 前端介面)
    B -- "PIL.Image (RGB)" --> C{YOLOv8 推論引擎}
    P[使用者參數] -- "IOU (float) / Conf (float)" --> C
    C -- "Boxes, Classes, Speed (Tensor)" --> D(OpenCV 標註與色彩轉換)
    D -- "Numpy Array (RGB)" --> E[Streamlit 顯示結果與文字]
```

### MSC (Message Sequence Chart - 訊息循序圖)
```mermaid
sequenceDiagram
    participant User as 使用者
    participant UI as Streamlit UI
    participant Model as YOLOv8 模型
    User->>UI: 1. 調整 IOU 與 Conf Threshold
    User->>UI: 2. 上傳測試圖片 (JPEG/PNG)
    UI->>Model: 3. predict_image(image, conf, iou)
    Note over Model: 執行神經網路推論<br/>與 NMS 過濾
    Model-->>UI: 4. 回傳偵測物件列表與處理時間
    UI-->>User: 5. 渲染 Bounding Box 圖片與偵測數據
```

### API Table
| API Function | Input Parameters | Data Type | Output / Return | Description |
| :--- | :--- | :--- | :--- | :--- |
| `load_model` | `model_name` | String | `ultralytics.YOLO` Object | 根據名稱動態載入 .pt 模型權重檔，並使用 `@st.cache_resource` 進行快取。 |
| `predict_image` | `model, image, conf_threshold, iou_threshold` | YOLO Object, PIL.Image, Float, Float | `Tuple[Numpy Array, String]` | 執行影像物件偵測，回傳疊加標註的 RGB 影像與格式化的預測結果字串。 |

---

## 4. 核心實作 (Coding)

專案的核心實作分為「訓練」與「推論」兩大部分。

### 模型訓練 (main.py)
呼叫 ultralytics API 進行模型從頭訓練，設定 150 epochs 與 imgsz 640。

```python
from ultralytics import YOLO

# 載入基礎模型架構
model = YOLO("yolov8n.yaml")  
model.to('cuda') # 啟用 GPU 加速

if __name__ == "__main__":
    # 訓練模型參數設定
    model.train(
        data="config.yaml",
        epochs=150,
        imgsz=640,
        patience=15,
        batch=16,
        workers=6
    )
```

---

## 5. 驗證 (Verification)

### 訓練指標驗證
經過 150 Epochs 的訓練，模型 Loss 持續下降且 Precision 穩步提升。YOLOv8 Small 相比於 Nano 在各項指標上表現出微幅領先。

### 測試集表現 (Croatia Fire Dataset)
* **Good Predictions (True Positives)**：兩個模型在大多數清晰的火災與煙霧場景下皆能成功標示物件。
* **Mixed Predictions (邊界案例)**：在某些背景複雜或輕微煙霧的場景中，出現了差異。Nano 模型容易出現 False Negative（漏判煙霧），而 Small 模型依然能保持良好的偵測能力（True Positive）。

---

### 成果展示 (Showcase)

### Training Results
Both models were trained for 150 epochs.
<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
    <img src="assets/results_nano.png" alt="Nano model training results" style="width: 45%; margin: 5px;">
    <img src="assets/results_small.png" alt="Small model training results" style="width: 45%; margin: 5px;">
</div>
<p align="center"><i>Fig 1. Comparison of Training Metrics (Loss, Precision, mAP) between Nano and Small models over 150 epochs.</i></p>

### Good predictions
Both models have shown great performance on most of the tested images.
<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
    <img src="assets/smoke_fire_true_positive_nano.jfif" alt="Nano model predictions" style="width: 45%; margin: 5px;">
    <img src="assets/smoke_fire_true_positive_small.jfif" alt="Small model predictions" style="width: 45%; margin: 5px;">
</div>
<p align="center"><i>Fig 2. Visualization of True Positive detections for both models in high-visibility fire and smoke scenarios.</i></p>

*While both models performed well, model based on YOLOv8s usually predicts with more precision and higher confidence levels.*

### Mixed predictions
Some predictions which resulted in different outcomes between the models.
<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
    <img src="assets/smoke_false_negative_nano.jfif" alt="Nano model predictions" style="width: 45%; margin: 5px;">
    <img src="assets/smoke_false_negative_small.jfif" alt="Small model predictions" style="width: 45%; margin: 5px;">
</div>
<p align="center"><i>Fig 3. Edge Case Analysis: Comparative performance on challenging low-contrast smoke patterns (Nano failing vs. Small succeeding).</i></p>

---

### 部署 (Deployment)

To demonstrate the capabilities of the trained models, Streamlit was used to create a simple, practical, and aesthetically pleasing web application.

🚀 **Try it out here**: [Fire and Smoke Detection App](https://fire-detection-yolo.streamlit.app/)

<div align="center">
  <img src="assets/app_test.PNG" alt="Streamlit app showcase" style="width: 80%; max-width: 800px;">
  <p><i>Fig 4. Full-scale application deployment showcase featuring the Streamlit web interface and real-time detection output.</i></p>
</div>
