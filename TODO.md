5/13 
1. #分析 改為 方塊分支圖(Breakdown) 要注意 方塊名稱
 與  DFD 中的名稱 有呼應，(方塊名稱可以是 cv.function() )
2. 改良 所有的 Fig caption


1. 實做 方案 B：實作 OpenCV 影像循環 (推薦給 Raspi)  2. 以剛剛的影片 測試





1. 輸出的影片 左上角要秀 FPS ，請改code
2. @README.md ， #需求 標記 assets/videos/roomfire41.mp4 來自          
   https://www.kaggle.com/datasets/unidpro/fire-and-smoke-dataset?resource=download
3. @README.md assets/videos/output_roomfire41.mp4 追加到 #驗證
4.  以 "source .venv/bin/activate && python3 process_enhanced_video." 執行



@README.md
1. 並行化流水線 可以容進  dataflow diagram
3. 
@README.md #分析 追加 針對  (INT8) 的 Quantininzation 原理描述 ，以表列解釋


4. 多測幾張datasetʼ統計，針對 yolov8n 處理結果 挑幾張 有失誤的 放在 #驗證
