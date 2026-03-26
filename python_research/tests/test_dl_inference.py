import pytest
import os
import cv2
import numpy as np
import sys

# 取得專案路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PR_ROOT = os.path.dirname(SCRIPT_DIR)
# 確保在 python 直接執行檔案時能找到 src
sys.path.insert(0, os.path.join(PR_ROOT, "src"))

from dl_inference_v1 import PCBDeepClassifier

@pytest.fixture(scope="module")
def classifier():
    """初始化分類器 - 使用測試專用的穩定模型 (Fixture)"""
    test_model_path = os.path.join(SCRIPT_DIR, "data", "models", "pcb_classifier_v1.onnx")
    
    if not os.path.exists(test_model_path):
        pytest.skip(f"Test model fixture not found at {test_model_path}")
        
    return PCBDeepClassifier(model_path=test_model_path)

@pytest.fixture(scope="module")
def test_img_path():
    """取得測試圖路徑"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "data", "fixtures", "sample_test.jpg")
    if not os.path.exists(path):
         pytest.skip(f"Local fixture not found: {path}")
    return path

def test_model_loading(classifier):
    """驗證模型與類別定義是否正確加載"""
    assert classifier.compiled_model is not None
    assert len(classifier.classes) == 7
    assert "Open" in classifier.classes

def test_single_inference_logic(classifier, test_img_path):
    """驗證單次推論流程與輸出格式"""
    img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    dummy_roi = [100, 100, 164, 164] # 64x64 area
    
    cls_id, conf = classifier.predict(img, dummy_roi)
    
    assert isinstance(cls_id, (int, np.integer))
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0

def test_roi_corner_sampling_robustness(classifier):
    """驗證邊緣 ROI 採樣強健性 (確保永遠輸出 64x64)"""
    dummy_img = np.zeros((100, 100), dtype=np.uint8)
    roi_corner = [0, 0, 10, 10]
    
    patch_norm = classifier.preprocess_roi(dummy_img, roi_corner)
    
    # 輸出 shape 應為 (1, 1, 64, 64) [batch, channel, h, w]
    assert patch_norm.shape == (1, 1, 64, 64)
    assert patch_norm.dtype == np.float32

if __name__ == "__main__":
    pytest.main([__file__])
