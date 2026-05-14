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
* **管理**：使用 **uv** 管理的虛擬環境來執行。
* **硬體**：
    * **開發與訓練**：Nvidia RTX 3070 Ti (CUDA)。
    * **部署與推論**：支援 Raspberry Pi 4 (CPU) 或其他邊緣運算裝置。
* **界面**：採用 Streamlit 構建的 Web UI。

### 邊緣運算優化 (Edge AI Optimization)
針對 **Raspberry Pi 4** 等效能有限的裝置，本專案提供以下優化措施：
* **尺寸優化**：建議將 `imgsz` 調降至 **320**，以平衡推論延遲與精確度。
* **模型轉換**：支援轉換為 **NCNN** (Next Generation CNN) 格式。
    * **原理簡述**：NCNN 是騰訊開發的高效能神經網路推論框架，無第三方依賴且針對 ARM 平台（如 NEON 指令集）進行深度優化。透過多核並行運算、記憶體池複用機制與運算子融合（Operator Fusion），顯著提升在 ARM CPU 上的推論效率。

### 界面
* **檔案輸入 (File Input)**：支援從本機上傳圖片（jpg, jpeg, png）。

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

**測試影片來源**：  
`assets/videos/roomfire41.mp4` 來自 [Kaggle - Fire and Smoke Dataset](https://www.kaggle.com/datasets/unidpro/fire-and-smoke-dataset?resource=download)。

---

## 2. 分析 (Analysis)

### 系統模組架構 (System Breakdown)
下圖展示了系統的模組化拆解，並對應 DFD 中的資料處理流程：

```mermaid
graph LR
    System[Fire Detection System]

    System --> UI[Frontend: Streamlit / CLI]
    UI --> UI1["cv2.VideoCapture() (Reader Thread)"]
    UI --> UI2["CLAHE + Gamma Enhancement (Worker Thread) LUT"]

    System --> Core[Inference: NCNN INT8 Engine]
    Core --> Core1["model.predict(imgsz=320, task='detect')"]

    System --> Post[Post-processing: OpenCV / Numpy]
    Post --> Post1["results[0].plot() (Bounding Boxes)"]
    Post --> Post2["cv2.putText() (NCNN FPS Overlay)"]
    Post --> Post3["cv2.VideoWriter() (Writer Thread)"]
```

### INT8 量化原理 (Quantization Principles)
針對邊緣運算裝置（如 Raspberry Pi 4），INT8 量化是提升推論速度的核心技術，其原理與效益如下表所示：

| 優化維度 (Dimension) | 原理描述 (Mechanism) | 效能效益 (Benefit) |
| :--- | :--- | :--- |
| **數值映射 (Mapping)** | 將模型權重從 32-bit 浮點數 (FP32) 映射至 8-bit 整數 (INT8) 空間，透過 Scaling Factor 與 Zero-point 進行線性轉換。 | **空間縮減**：減少 75% 的模型權重體積與記憶體佔用，有利於快取命中。 |
| **運算加速 (Acceleration)** | 利用 ARM CPU 的 SIMD (如 NEON 指令集) 進行整數並行運算，取代耗時的浮點數運算。 | **速度提升**：在非 GPU 裝置上，整數運算吞吐量遠高於浮點運算，顯著提升 FPS。 |
| **頻寬優化 (Bandwidth)** | 降低資料在 CPU 與記憶體（DRAM）之間傳輸所需的位元寬度。 | **降低延遲**：減少記憶體存取瓶頸（Memory Bound），提升系統整體的響應速度。 |

---

## 3. 設計 (Design)

### Data Flow Diagram (資料流圖)
```mermaid
graph TD
    A[使用者輸入源] -- "Image/Video Frame (BGR)" --> B(影像增強: Worker Thread + LUT)
    B -- "CLAHE + Gamma (BGR)" --> C(推論引擎: NCNN INT8)
    P[使用者參數] -- "IOU / Conf / imgsz=320" --> C
    C -- "Detection Results" --> D(後處理: OpenCV / Numpy)
    D -- "Processed Frame (RGB/BGR)" --> E[前端界面: Streamlit / CLI]
```

### MSC (Message Sequence Chart - 訊息循序圖)
```mermaid
sequenceDiagram
    participant User as 使用者
    participant UI as Streamlit / Script
    participant Proc as 影像增強 (OpenCV)
    participant Model as YOLOv8 模型
    User->>UI: 1. 設定參數與輸入源
    UI->>Proc: 2. 執行 CLAHE 與 Gamma 修正
    Proc-->>UI: 3. 回傳增強後的影像幀
    UI->>Model: 4. 執行 model.predict(device='cpu')
    Note over Model: 執行神經網路推論<br/>與 NMS 過濾
    Model-->>UI: 5. 回傳偵測物件列表
    UI-->>User: 6. 渲染 FPS、標註框並輸出結果
```

### API Table
| API Function | Input Parameters | Data Type | Output / Return | Description |
| :--- | :--- | :--- | :--- | :--- |
| `load_model` | `model_name` | String | `ultralytics.YOLO` Object | 根據名稱動態載入 .pt 模型權重檔，並使用 `@st.cache_resource` 進行快取。 |
| `predict_image` | `model, image, conf_threshold, iou_threshold` | YOLO Object, PIL.Image, Float, Float | `Tuple[Numpy Array, String]` | 執行影像物件偵測，回傳疊加標註的 RGB 影像與格式化的預測結果字串。 |

---


## 4. 驗證 (Verification)

### 訓練指標驗證
經過 150 Epochs 的訓練，模型 Loss 持續下降且 Precision 穩步提升。YOLOv8 Small 相比於 Nano 在各項指標上表現出微幅領先。

### 測試集表現 (Croatia Fire Dataset)
* **Good Predictions (True Positives)**：兩個模型在大多數清晰的火災與煙霧場景下皆能成功標示物件。
* **Mixed Predictions (邊界案例)**：在某些背景複雜或輕微煙霧的場景中，出現了差異。Nano 模型容易出現 False Negative（漏判煙霧），而 Small 模型依然能保持良好的偵測能力（True Positive）。
* **影片測試結果**：可參考 `assets/videos/output_roomfire41.mp4` 觀察在動態場景下的偵測穩定度與 FPS 表現。

---

### 成果展示 (Showcase)

### Training Results
Both models were trained for 150 epochs.
<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
    <img src="assets/images/results_nano.png" alt="Nano model training results" style="width: 45%; margin: 5px;">
    <img src="assets/images/results_small.png" alt="Small model training results" style="width: 45%; margin: 5px;">
</div>
<p align="center"><i>Fig 1. Comparison of Training Metrics (Loss, Precision, mAP) between Nano and Small models over 150 epochs.</i></p>

### Good predictions
Both models have shown great performance on most of the tested images.
<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
    <img src="assets/images/smoke_fire_true_positive_nano.jfif" alt="Nano model predictions" style="width: 45%; margin: 5px;">
    <img src="assets/images/smoke_fire_true_positive_small.jfif" alt="Small model predictions" style="width: 45%; margin: 5px;">
</div>
<p align="center"><i>Fig 2. Visualization of True Positive detections for both models in high-visibility fire and smoke scenarios.</i></p>

*While both models performed well, model based on YOLOv8s usually predicts with more precision and higher confidence levels.*

### Mixed predictions
Some predictions which resulted in different outcomes between the models.
<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
    <img src="assets/images/smoke_false_negative_nano.jfif" alt="Nano model predictions" style="width: 45%; margin: 5px;">
    <img src="assets/images/smoke_false_negative_small.jfif" alt="Small model predictions" style="width: 45%; margin: 5px;">
</div>
<p align="center"><i>Fig 3. Edge Case Analysis: Comparative performance on challenging low-contrast smoke patterns (Nano failing vs. Small succeeding).</i></p>

### 以下列指令執行離線影片增強偵測：
```bash
source .venv/bin/activate && python3 process_enhanced_video.py
```
`assets/videos/output_roomfire41.mp4` 即為處理後的成果。

