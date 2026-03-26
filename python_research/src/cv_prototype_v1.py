import cv2
import numpy as np
import time
import os

class PCBVisionPrototype:
    def __init__(self, debug=True):
        self.debug = debug
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def process(self, template_path, test_path):
        """
        執行參考比較法定位瑕疵 ROI
        """
        metrics = {}
        
        # 1. 載入圖像
        start_time = time.time()
        img_temp = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        img_test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        metrics['load_time_ms'] = (time.time() - start_time) * 1000

        if img_temp is None or img_test is None:
            raise FileNotFoundError("無法載入圖像，請檢查路徑。")

        # 2. 差異檢測 (Abs Diff)
        start_time = time.time()
        diff = cv2.absdiff(img_temp, img_test)
        metrics['absdiff_time_ms'] = (time.time() - start_time) * 1000

        # 3. 二值化 (Thresholding)
        start_time = time.time()
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        metrics['threshold_time_ms'] = (time.time() - start_time) * 1000

        # 4. 形態學優化 (Morphology)
        start_time = time.time()
        # 使用 Closing 連接微小斷點，使用 Opening 去除椒鹽雜訊
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, self.kernel)
        metrics['morphology_time_ms'] = (time.time() - start_time) * 1000

        # 5. 連通域分析 (CCL) 萃取 ROI
        start_time = time.time()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed, connectivity=8)
        
        rois = []
        for i in range(1, num_labels): # 0 是背景
            x, y, w, h, area = stats[i]
            if area > 10: # 過濾極小雜訊
                rois.append([x, y, x + w, y + h]) # 符合 detection_schema.md [x1, y1, x2, y2]
        
        metrics['ccl_time_ms'] = (time.time() - start_time) * 1000
        metrics['total_latency_ms'] = sum(metrics.values())
        metrics['num_rois'] = len(rois)

        return rois, metrics


