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

from cv_prototype_v1 import PCBVisionPrototype

@pytest.fixture(scope="module")
def detector():
    """初始化 detector 並關閉 debug 模式"""
    return PCBVisionPrototype(debug=False)

@pytest.fixture(scope="module")
def paths():
    """準備測試路徑與合成圖數據"""
    fixture_dir = os.path.join(SCRIPT_DIR, "data", "fixtures")
    temp_path = os.path.join(fixture_dir, "sample_temp.jpg")
    test_path = os.path.join(fixture_dir, "sample_test.jpg")
    
    if not os.path.exists(temp_path):
        os.makedirs(fixture_dir, exist_ok=True)
        # 生成基準圖 (128 灰度)
        base = np.full((100, 100), 128, dtype=np.uint8)
        cv2.imwrite(temp_path, base)
        # 生成測試圖 (15x15 的瑕疵)
        test = base.copy()
        cv2.rectangle(test, (70, 70), (85, 85), 255, -1)
        cv2.imwrite(test_path, test)
        
    return {"temp": temp_path, "test": test_path, "fixture_dir": fixture_dir}

def test_determinism(detector, paths):
    """[確定性] 驗證重複執行相同輸入，ROI 座標必須完全一致"""
    res1, _ = detector.process(paths["temp"], paths["test"])
    res2, _ = detector.process(paths["temp"], paths["test"])
    assert res1 == res2, "CV output is non-deterministic on identical runs!"

def test_isolated_standard_case(detector, paths):
    """[規格測試] 驗證標準瑕疵應被正確定位"""
    rois, _ = detector.process(paths["temp"], paths["test"])
    assert len(rois) > 0, "Failed to detect standard fixtures/synthetic defect!"
    
    for roi in rois:
        assert len(roi) == 4, "ROI structure invalid!"
        assert roi[0] < roi[2], "ROI X-order invalid!"
        assert roi[1] < roi[3], "ROI Y-order invalid!"

def test_edge_case_identical_images(detector, paths):
    """[邊界項] 若兩圖完全相同，不應偵測到任何 ROI"""
    rois, _ = detector.process(paths["temp"], paths["temp"])
    assert len(rois) == 0, "Identical images should yield 0 ROIs!"

def test_edge_case_small_noise(detector, paths):
    """[邏輯邊界] 驗證極小面積雜訊 (如 1x1 像素) 是否被過濾"""
    noise_img_path = os.path.join(paths["fixture_dir"], "noise_tmp.jpg")
    noise_img = cv2.imread(paths["temp"], cv2.IMREAD_GRAYSCALE)
    cv2.rectangle(noise_img, (10, 10), (11, 11), 255, -1)
    cv2.imwrite(noise_img_path, noise_img)
    
    try:
        rois, _ = detector.process(paths["temp"], noise_img_path)
        # 依據 area > 10 邏輯，4 像素的點應該被過濾
        assert len(rois) == 0, "Small noise area(4) should be filtered out by Area threshold(10)!"
    finally:
        if os.path.exists(noise_img_path):
            os.remove(noise_img_path)

def test_error_invalid_path(detector):
    """[錯誤處理] 驗證路徑錯誤時會拋出合理異常"""
    with pytest.raises(FileNotFoundError):
        detector.process("invalid_a.jpg", "invalid_b.jpg")

if __name__ == "__main__":
    pytest.main([__file__])
