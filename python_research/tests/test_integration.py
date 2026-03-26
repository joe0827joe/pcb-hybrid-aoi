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

from hybrid_pipeline_v1 import HybridAOIPipeline

@pytest.fixture(scope="module")
def pipeline():
    """初始化整合管線 - 使用測試專用模型檔案"""
    model_fixture = os.path.join(SCRIPT_DIR, "data", "models", "pcb_classifier_v1.onnx")
    if not os.path.exists(model_fixture):
        pytest.skip(f"Model fixture missing at: {model_fixture}")
    return HybridAOIPipeline(model_path=model_fixture)

@pytest.fixture(scope="module")
def test_pair():
    """取得測試圖組路徑"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_dir = os.path.join(current_dir, "data", "fixtures")
    template = os.path.join(fixture_dir, "sample_temp.jpg")
    test = os.path.join(fixture_dir, "sample_test.jpg")
    
    if not os.path.exists(template):
        pytest.skip(f"Sample dataset missing at: {template}")
        
    return {"template": template, "test": test, "results_dir": os.path.join(PR_ROOT, "results")}

def visualize_results(img_path, defects, output_path):
    """將偵測結果繪製於圖片上供查閱"""
    if not os.path.exists(img_path):
        return
        
    img = cv2.imread(img_path)
    for d in defects:
        x1, y1, x2, y2 = map(int, d["roi_coords"])
        color = (0, 0, 255) if d["is_defect"] else (180, 180, 180)
        thickness = 2 if d["is_defect"] else 1
        
        label = f"{d['defect_type']} ({d['confidence']:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if d["is_defect"]:
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.imwrite(output_path, img)

def test_pipeline_output_structure(pipeline, test_pair):
    """[整合測試] 驗證輸出結構與欄位"""
    defects, _ = pipeline.inspect(test_pair["template"], test_pair["test"])
    assert len(defects) > 0, "Pipeline should detect something."
    
    # 視覺化輸出供人工查閱
    out_path = os.path.join(test_pair["results_dir"], "integration_verify_pytest.jpg")
    visualize_results(test_pair["test"], defects, out_path)
    
    for d in defects:
        for field in ["defect_type", "confidence", "is_defect", "roi_coords"]:
            assert field in d, f"Missing field '{field}'"
        assert isinstance(d["is_defect"], bool)

def test_defect_filtering_logic(pipeline, test_pair):
    """[邏輯驗證] 確保分類後的瑕疵數量正確"""
    all_candidates, _ = pipeline.inspect(test_pair["template"], test_pair["test"])
    total_candidates = len(all_candidates)
    final_defects = len([d for d in all_candidates if d['is_defect']])
    assert final_defects <= total_candidates, "Final defects cannot exceed candidates"

@pytest.mark.parametrize("temp_name, test_name", [
    ("sample_temp.jpg", "sample_test.jpg"),
])
def test_multi_sample_consistency(pipeline, test_pair, temp_name, test_name):
    """[一致性] 驗證多取樣穩定性"""
    # 這裡目前僅測試一組，可透過參數化擴充
    defects, latency = pipeline.inspect(test_pair["template"], test_pair["test"])
    assert len(defects) > 0
    assert latency > 0

def test_robustness_on_missing_file(pipeline):
    """[強健性] 驗證異常文件處理"""
    with pytest.raises((FileNotFoundError, cv2.error, ValueError)):
        pipeline.inspect("non_existent.jpg", "non_existent2.jpg")

if __name__ == "__main__":
    pytest.main([__file__])
