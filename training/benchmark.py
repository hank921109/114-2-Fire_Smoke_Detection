from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import os

# 定義路徑
MODEL_PATH = "../Deployment/Models/nano.pt"
DATA_PATH = "config.yaml"

if __name__ == "__main__":
    # 1. 執行效能測試 (Benchmark)
    print("--- 開始執行 Raspi 效能測試 (CPU) ---")
    # 注意：在 Raspi 上效能較低，只需測試核心格式即可
    benchmark(
        model=MODEL_PATH,
        data=DATA_PATH,
        device="cpu",      # Raspi 4 使用 CPU
        imgsz=320,         # 建議縮小尺寸以符合 Raspi 效能 (原 640)
        half=False,        # CPU 不支援半精度 (FP16)
        int8=False         # 如果有量化需求可開啟
    )

    # 2. 模型轉換 (Export to NCNN)
    print("\n--- 開始轉換模型為 NCNN 格式 ---")
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        # 轉換為 NCNN 格式，適合 ARM CPU 加速
        model.export(format="ncnn", imgsz=320)
        print(f"轉換完成！NCNN 模型已儲存在 {MODEL_PATH.replace('.pt', '_ncnn')}")
    else:
        print(f"錯誤：找不到模型文件 {MODEL_PATH}")