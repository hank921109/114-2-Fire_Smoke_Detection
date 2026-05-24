~~5/13~~ 
1. ~~#分析 改為 方塊分支圖(Breakdown) 要注意 方塊名稱 與 DFD 中的名稱 有呼應，(方塊名稱可以是 cv.function() )~~
2. ~~改良 所有的 Fig caption~~

~~5/15~~
1. ~~實做 方案 B：實作 OpenCV 影像循環 (推薦給 Raspi)~~
2. ~~以剛剛的影片 測試~~

~~5/18~~
1. ~~輸出的影片 左上角要秀 FPS ，請改code~~
2. ~~@README.md ， #需求 標記 assets/videos/roomfire41.mp4 來自 https://www.kaggle.com/datasets/unidpro/fire-and-smoke-dataset?resource=download~~
3. ~~@README.md assets/videos/output_roomfire41.mp4 追加到 #驗證~~
4. ~~以 "source .venv/bin/activate && python3 process_enhanced_video." 執行~~

~~5/19~~
1. ~~NCNN 要(全名)ʼ簡述原理~~
2. ~~DFD 中的 Yolo 不該為 菱形，請修正~~
3. ~~檢查~~

~~5/20~~
1. ~~並行化流水線 可以容進 dataflow diagram~~
3. 
4. ~~@README.md #分析 追加 針對 (INT8) 的 Quantininzation 原理描述 ，以表列解釋~~
5. ~~多測幾張datasetʼ統計，針對 yolov8n 處理結果 挑幾張 有失誤的 放在 #驗證~~

5/24
1. 構思加快 FPS 的方法 (例如：跳幀處理、降低解析度、NCNN Vulkan 加速)
2. 檢查 Breakdown 的模組名稱、Dataflow 的模組名稱、API table 的模組名稱，確保三者完全一致
3. 創建 "pipeline" 資料夾，並修改程式碼儲存 Dataflow 各階段的中間結果 (如：CLAHE, Gamma, YOLO 偵測結果)
4. 將 NCNN CPU 方案 切換為 TensorRT GPU 方案 (針對 Jetson Orin Nano 優化)
