import cv2
import time
from cv_prototype_v1 import PCBVisionPrototype
from dl_inference_v1 import PCBDeepClassifier

class HybridAOIPipeline:
    def __init__(self):
        # 初始化 Stage 1: CV 定位器 (極速定位)
        self.stage1 = PCBVisionPrototype()
        # 初始化 Stage 2: DL 分類器 (精準濾噪)
        self.stage2 = PCBDeepClassifier()
        
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

if __name__ == "__main__":
    SAMPLE_TEMP = "data/DeepPCB-master/PCBData/group00041/00041/00041000_temp.jpg"
    SAMPLE_TEST = "data/DeepPCB-master/PCBData/group00041/00041/00041000_test.jpg"
    
    pipeline = HybridAOIPipeline()
    defects, latency = pipeline.inspect(SAMPLE_TEMP, SAMPLE_TEST)
    
    print(f"--- [HYBRID PIPELINE] Inspection Report ---")
    print(f"Total Latency: {latency:.2f} ms")
    print(f"Total ROIs from Stage 1: {len(defects)}")
    
    confirmed_defects = [d for d in defects if d['is_defect']]
    print(f"Confirmed Defects (Stage 2): {len(confirmed_defects)}")
    
    for d in confirmed_defects:
        print(f"  > Found {d['defect_type']} at {d['roi_coords']} | Conf: {d['confidence']:.2f}")
    
    # 效能門檻檢查 (AGENTRULE.md: < 130ms)
    if latency < 130:
        print(f"✅ Performance Gate: PASSED (Under 130ms)")
    else:
        print(f"❌ Performance Gate: FAILED (Exceeded 130ms)")
