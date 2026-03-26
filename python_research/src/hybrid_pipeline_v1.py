import cv2
import time
from cv_prototype_v1 import PCBVisionPrototype
from dl_inference_v1 import PCBDeepClassifier

class HybridAOIPipeline:
    def __init__(self, model_path=None):
        # 初始化 Stage 1: CV 定位器 (極速定位)
        self.stage1 = PCBVisionPrototype()
        # 初始化 Stage 2: DL 分類器 (精準濾噪)
        # [Ambient-Independent] 支持指定測試專用模型路徑
        self.stage2 = PCBDeepClassifier(model_path=model_path)
        
        # 標註類別對齊 detection_schema.md
        self.classes = ["Background", "Open", "Short", "Mousebite", "Spur", "Copper", "Pin-hole"]

    def inspect(self, template_path, test_path):
        """
        全自動二段式檢測
        """
        results = []
        overall_start = time.time()
        
        # --- Stage 1: CV 定位 ---
        rois, s1_metrics = self.stage1.process(template_path, test_path)
        
        # --- Stage 2: 批次推論與濾噪 ---
        img_test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        
        # 2.1 集中預處理
        patches = []
        for roi in rois:
            patches.append(self.stage2.preprocess_roi(img_test, roi))
            
        # 2.2 一次性 Batch 推論 (模擬 OpenVINO 開發者模式)
        batch_results = self.stage2.predict_batch(patches)
        
        for idx, (roi, (cls_id, confidence)) in enumerate(zip(rois, batch_results)):
            results.append({
                "roi_id": idx,
                "roi_coords": roi,
                "defect_type": self.classes[cls_id],
                "confidence": confidence,
                "is_defect": (cls_id > 0) and (confidence > 0.85) # 濾除非瑕疵類別
            })

        total_latency = (time.time() - overall_start) * 1000
        
        return results, total_latency


