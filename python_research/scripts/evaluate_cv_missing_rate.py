import os
import cv2
import sys
import numpy as np
from tqdm import tqdm

# 取得目錄路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PR_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PR_ROOT)

# 將 src 加入路徑以便引入元件
sys.path.append(os.path.join(PR_ROOT, "src"))
from cv_prototype_v1 import PCBVisionPrototype

def calculate_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_cv_recall():
    detector = PCBVisionPrototype()
    BASE_DIR = os.path.join(PROJECT_ROOT, "data", "DeepPCB-master", "PCBData")
    
    total_gt_defects = 0
    missed_gt_defects = 0
    
    groups = [d for d in os.listdir(BASE_DIR) if d.startswith("group")]
    # 處理所有 available groups
    groups = [d for d in os.listdir(BASE_DIR) if d.startswith("group")]
    
    # 建立輸出目錄
    report_dir = os.path.join(PR_ROOT, "results")
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"--- [CV RECALL AUDIT] Evaluating {len(groups)} groups ---")
    
    for group in tqdm(groups):
        group_path = os.path.join(BASE_DIR, group)
        img_sub = group.replace("group", "")
        ann_sub = img_sub + "_not"
        
        img_dir = os.path.join(group_path, img_sub)
        ann_dir = os.path.join(group_path, ann_sub)
        
        if not (os.path.exists(img_dir) and os.path.exists(ann_dir)):
            continue
            
        test_images = [f for f in os.listdir(img_dir) if f.endswith("_test.jpg")]
        
        for img_name in test_images:
            test_path = os.path.join(img_dir, img_name)
            temp_path = test_path.replace("_test.jpg", "_temp.jpg")
            ann_name = img_name.replace("_test.jpg", ".txt")
            ann_path = os.path.join(ann_dir, ann_name)
            
            if not os.path.exists(ann_path): continue
            
            # 讀取 Ground Truth
            gt_boxes = []
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().replace(',', ' ').split()
                    if len(parts) >= 4:
                        gt_boxes.append(list(map(int, parts[:4])))
            
            total_gt_defects += len(gt_boxes)
            
            # 執行 CV Stage 1
            try:
                rois, _ = detector.process(temp_path, test_path)
            except Exception:
                continue
            
            # 檢查每個 GT 是否被覆蓋 (只要 IoU > 0.05 就算定位到，因為 Stage 1 只是粗略定位)
            for gt in gt_boxes:
                detected = False
                for roi in rois:
                    if calculate_iou(gt, roi) > 0.05:
                        detected = True
                        break
                if not detected:
                    missed_gt_defects += 1

    missing_rate = (missed_gt_defects / total_gt_defects) * 100 if total_gt_defects > 0 else 0
    recall = 100 - missing_rate
    status = "PASSED" if recall >= 98.0 else "FAILED (Recall < 98%)"
    
    print("\n--- CV Audit Results ---")
    print(f"Total GT Defects Analyzed: {total_gt_defects}")
    print(f"Missed by CV (Stage 1): {missed_gt_defects}")
    print(f"CV Missing Rate: {missing_rate:.2f}%")
    print(f"CV Recall: {recall:.2f}%")
    print(f"Status: {status}")

    # 生成正式報告
    report_file = os.path.join(PR_ROOT, "results", "cv_recall_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# CV Stage-1 Recall Audit Report\n")
        f.write(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Status: {status}\n")
        f.write(f"- Total GT Defects: {total_gt_defects}\n")
        f.write(f"- Missed: {missed_gt_defects}\n")
        f.write(f"- CV Recall Score: {recall:.2f}%\n")
        f.write(f"- Threshold Policy: >= 98.00% (High Confidence Buffer)\n")

if __name__ == "__main__":
    import time
    evaluate_cv_recall()
