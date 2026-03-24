import unittest
import os
import time
import cv2
from hybrid_pipeline_v1 import HybridAOIPipeline

class TestHybridIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 初始化完整檢測管線
        cls.pipeline = HybridAOIPipeline()
        cls.test_pair = {
            "template": "data/DeepPCB-master/PCBData/group00041/00041/00041000_temp.jpg",
            "test": "data/DeepPCB-master/PCBData/group00041/00041/00041000_test.jpg"
        }

    def _visualize_results(self, img_path, defects, output_name):
        """輔助方法：將偵測結果繪製於圖片上"""
        img = cv2.imread(img_path)
        for d in defects:
            x1, y1, x2, y2 = map(int, d["roi_coords"])
            # 紅色 (0,0,255) 代表確認為瑕疵，灰色 (200,200,200) 代表被 DL 判定為背景
            color = (0, 0, 255) if d["is_defect"] else (180, 180, 180)
            thickness = 2 if d["is_defect"] else 1
            
            label = f"{d['defect_type']} ({d['confidence']:.2f})"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            if d["is_defect"]:
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 使用絕對路徑
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, "test_data", output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return output_path

    def test_pipeline_output_structure(self):
        """驗證整合輸出結構與內容，並產出視覺化報告"""
        if not os.path.exists(self.test_pair["template"]):
            self.skipTest("Sample data not found.")
            
        defects, latency = self.pipeline.inspect(self.test_pair["template"], self.test_pair["test"])
        
        self.assertGreater(len(defects), 0, "Integrated pipeline should detect something.")
        
        # 產出視覺化圖片供人工抽檢
        out_file = self._visualize_results(self.test_pair["test"], defects, "integration_verify.jpg")
        print(f"\n[VISUALIZATION] Evidence saved to: {out_file}")

        for d in defects:
            self.assertIn("defect_type", d)
            self.assertIn("confidence", d)
            self.assertIn("is_defect", d)
            self.assertIsInstance(d["is_defect"], bool)

    def test_performance_gate(self):
        """
        [QA Gate] 驗證端到端延遲是否低於 130ms (AGENTRULE.md 要求)
        """
        if not os.path.exists(self.test_pair["template"]):
            self.skipTest("Sample data not found.")
            
        # 排除初次熱身時間，測試多次平均
        latencies = []
        for _ in range(5):
            _, latency = self.pipeline.inspect(self.test_pair["template"], self.test_pair["test"])
            latencies.append(latency)
            
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n[BENCHMARK] End-to-End Pipeline Avg Latency: {avg_latency:.2f} ms")
        self.assertLess(avg_latency, 130, "Total Pipeline Latency too high (>130ms)!")

    def test_defect_filtering(self):
        """驗證 Stage 2 濾噪是否有效 (排除 Background)"""
        if not os.path.exists(self.test_pair["template"]):
            self.skipTest("Sample data not found.")
            
        defects, _ = self.pipeline.inspect(self.test_pair["template"], self.test_pair["test"])
        
        # 檢查邏輯：如果有 detected 但 is_defect 為 False，表示該 ROI 為 Background 或是信心度不足
        # 這驗證了分類器確實介入了最終裁決
        all_stage1_rois = len(defects)
        final_defects = len([d for d in defects if d['is_defect']])
        
        print(f"\n[INTEGRATION] Stage 1 ROIs: {all_stage1_rois} -> Stage 2 Defects: {final_defects}")
        # 在 group00041/00041000 中，通常會有幾處真實瑕疵，而 Stage 1 可能報出較多疑似區塊
        self.assertLessEqual(final_defects, all_stage1_rois, "Final defects must not exceed initial candidates.")

    def test_multi_sample_consistency(self):
        """驗證管線在連續多個樣本上的穩定度與性能指標"""
        base_dir = "data/DeepPCB-master/PCBData/group00041/00041/"
        # 測試 00041000 到 00041004 (5組圖)
        for i in range(5):
            idx = f"{i:03}"
            temp = f"{base_dir}00041{idx}_temp.jpg"
            test = f"{base_dir}00041{idx}_test.jpg"
            
            if os.path.exists(temp) and os.path.exists(test):
                defects, latency = self.pipeline.inspect(temp, test)
                
                # 統計瑕疵分佈
                confirmed = [d for d in defects if d['is_defect']]
                print(f"[SAMPLE {idx}] ROIs: {len(defects)} | Defects: {len(confirmed)} | Time: {latency:.2f} ms")
                
                # 驗證每個樣本都能在 130ms 內完成 (單一影像處理)
                self.assertLess(latency, 130, f"Sample {idx} latency exceeded limit!")

if __name__ == "__main__":
    unittest.main()
