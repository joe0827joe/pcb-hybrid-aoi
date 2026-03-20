import cv2
import os
from cv_prototype_v1 import PCBVisionPrototype

def run_visualization():
    SAMPLE_TEMP = "data/DeepPCB-master/PCBData/group00041/00041/00041000_temp.jpg"
    SAMPLE_TEST = "data/DeepPCB-master/PCBData/group00041/00041/00041000_test.jpg"
    OUTPUT_PATH = "python_research/test_data/result_visualization.jpg"

    detector = PCBVisionPrototype()
    # 1. 執行偵測
    rois, metrics = detector.process(SAMPLE_TEMP, SAMPLE_TEST)
    
    # 2. 讀取 Ground Truth 標註 (DeepPCB 格式: x1,y1,x2,y2,type)
    gt_path = SAMPLE_TEST.replace("_test.jpg", ".txt")
    gt_rois = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split() # DeepPCB txt 使用空格分隔或逗號
                if len(parts) >= 4:
                    gt_rois.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])

    # 3. 準備繪圖 (將灰階轉彩色以利標註)
    img_test = cv2.imread(SAMPLE_TEST)
    
    # 在測試圖上畫出偵測到的 ROI (紅框)
    for roi in rois:
        x1, y1, x2, y2 = roi
        cv2.rectangle(img_test, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_test, "Pred ROI", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 4. 在測試圖上畫出標註的 Ground Truth (綠框)
    for gt in gt_rois:
        gx1, gy1, gx2, gy2 = gt
        cv2.rectangle(img_test, (gx1, gy1), (gx2, gy2), (0, 255, 0), 1)
        cv2.putText(img_test, "GT", (gx2-20, gy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 存檔
    cv2.imwrite(OUTPUT_PATH, img_test)
    print(f"--- Detection Effect Analysis ---")
    print(f"✅ Ground Truth Defect Count: {len(gt_rois)}")
    print(f"✅ Algorithm Detected ROI Count: {len(rois)}")
    print(f"✅ Detection Results saved to: {OUTPUT_PATH}")
    print(f"--- Logic Summary ---")
    print(f"紅框 (Red) 為演算法鎖定的待分類區域，綠框 (Green) 為 DeepPCB 標註的真實瑕疵處。")
    print(f"若紅框覆蓋了所有綠框，即完成 CV 階段的關鍵任務：高召回率。")

if __name__ == "__main__":
    run_visualization()
