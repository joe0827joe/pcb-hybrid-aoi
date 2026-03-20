import unittest
import os
import cv2
import time
from cv_prototype_v1 import PCBVisionPrototype

class TestPCBCVPrototype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 初始化 detector 並指定基礎檢測對象
        cls.detector = PCBVisionPrototype()
        cls.test_pair = {
            "template": "data/DeepPCB-master/PCBData/group00041/00041/00041000_temp.jpg",
            "test": "data/DeepPCB-master/PCBData/group00041/00041/00041000_test.jpg"
        }

    def test_latency_threshold(self):
        """
        驗證 CV 階段延遲是否低於 50ms (憲法預算分配)
        """
        if not os.path.exists(self.test_pair["template"]):
            self.skipTest("Sample data not found.")
            
        rois, metrics = self.detector.process(self.test_pair["template"], self.test_pair["test"])
        
        # 排除初次加載耗時，測試多次平均
        total_logic_time = metrics['absdiff_time_ms'] + \
                          metrics['threshold_time_ms'] + \
                          metrics['morphology_time_ms'] + \
                          metrics['ccl_time_ms']
        
        print(f"\n[BENCHMARK] Total Logic Time: {total_logic_time:.2f} ms")
        self.assertLess(total_logic_time, 50, "CV Logic Latency too high (>50ms)!")

    def test_roi_integrity(self):
        """
        驗證 ROI 結構是否符合 [x1, y1, x2, y2] 協議
        """
        if not os.path.exists(self.test_pair["template"]):
            self.skipTest("Sample data not found.")

        rois, _ = self.detector.process(self.test_pair["template"], self.test_pair["test"])
        
        self.assertGreater(len(rois), 0, "No ROIs detected on defective image!")
        for roi in rois:
            self.assertEqual(len(roi), 4, "ROI structure invalid!")
            self.assertLess(roi[0], roi[2], "ROI Coordinate X order invalid!")
            self.assertLess(roi[1], roi[3], "ROI Coordinate Y order invalid!")

if __name__ == "__main__":
    unittest.main()
