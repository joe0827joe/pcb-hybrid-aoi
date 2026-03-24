import unittest
import os
import cv2
import numpy as np
import time
from dl_inference_v1 import PCBDeepClassifier

class TestPCBDLInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 初始化分類器
        try:
            cls.classifier = PCBDeepClassifier(model_path="python_research/pcb_classifier_v1.onnx")
        except Exception as e:
            raise unittest.SkipTest(f"Model file not found or OpenVINO error: {e}")
        
        cls.test_img_path = "data/DeepPCB-master/PCBData/group00041/00041/00041000_test.jpg"

    def test_model_loading(self):
        """驗證模型與類別定義是否正確加載"""
        self.assertIsNotNone(self.classifier.compiled_model)
        self.assertEqual(len(self.classifier.classes), 7)
        self.assertIn("Open", self.classifier.classes)

    def test_single_inference_logic(self):
        """驗證單次推論流程與輸出格式"""
        if not os.path.exists(self.test_img_path):
            self.skipTest("Sample image not found.")
            
        img = cv2.imread(self.test_img_path, cv2.IMREAD_GRAYSCALE)
        dummy_roi = [100, 100, 164, 164] # 64x64 area
        
        cls_id, conf = self.classifier.predict(img, dummy_roi)
        
        self.assertIsInstance(cls_id, (int, np.integer))
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_roi_corner_sampling_robustness(self):
        """驗證邊緣 ROI 採樣強健性 (確保永遠輸出 64x64)"""
        dummy_img = np.zeros((100, 100), dtype=np.uint8)
        # ROI 位於極邊緣 (0, 0)
        roi_corner = [0, 0, 10, 10]
        
        patch_norm = self.classifier.preprocess_roi(dummy_img, roi_corner)
        
        # 輸出 shape 應為 (1, 1, 64, 64) [batch, channel, h, w]
        self.assertEqual(patch_norm.shape, (1, 1, 64, 64))
        self.assertEqual(patch_norm.dtype, np.float32)

    def test_batch_inference_performance(self):
        """
        驗證批次推論性能 (目標: 10個 ROI 應在 50ms 內完成推論)
        """
        if not os.path.exists(self.test_img_path):
            self.skipTest("Sample image not found.")
            
        img = cv2.imread(self.test_img_path, cv2.IMREAD_GRAYSCALE)
        dummy_roi = [100, 100, 164, 164]
        
        # 準備 10 個相同的 patches 測試 Batch 效能
        patch = self.classifier.preprocess_roi(img, dummy_roi)
        patches = [patch] * 10
        
        start_time = time.time()
        results = self.classifier.predict_batch(patches)
        latency = (time.time() - start_time) * 1000
        
        print(f"\n[BENCHMARK] Batch (10 ROIs) Inference Latency: {latency:.2f} ms")
        self.assertEqual(len(results), 10)
        # 門檻設定：10個 patch 應該在 50ms 內 (OpenVINO CPU 通常更前端)
        self.assertLess(latency, 50, "DL Batch Inference too slow (>50ms)!")

if __name__ == "__main__":
    unittest.main()
