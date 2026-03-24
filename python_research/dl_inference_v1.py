import cv2
import numpy as np
import time
import openvino.runtime as ov

class PCBDeepClassifier:
    def __init__(self, model_path="python_research/pcb_classifier_v1.onnx", device='CPU'):
        print(f"--- [DL Engine] Loading Real Model: {model_path} ---")
        try:
            core = ov.Core()
            model = core.read_model(model_path)
            self.compiled_model = core.compile_model(model, device)
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            self.classes = ["Background", "Open", "Short", "Mousebite", "Spur", "Copper", "Pin-hole"]
        except Exception as e:
            print(f"❌ Error loading OpenVINO: {e}")
            raise

    def preprocess_roi(self, full_image, roi):
        """
        [v2 Fix] 採用同尺度固定窗口採樣 (Fixed-Scale Window)
        避免因為 Resize 導致的瑕疵特徵扭曲
        """
        x1, y1, x2, y2 = map(int, roi)
        h_max, w_max = full_image.shape[:2]
        
        # 1. 計算幾何中心點
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # 2. 定義 64x64 採樣區間 (half_size = 32)
        px1, py1 = cx - 32, cy - 32
        px2, py2 = cx + 32, cy + 32
        
        # 3. 邊界溢出保護與滑動校正 (確保永遠採集到 64x64)
        if px1 < 0:
            px2 += abs(px1)
            px1 = 0
        if py1 < 0:
            py2 += abs(py1)
            py1 = 0
        if px2 > w_max:
            px1 -= (px2 - w_max)
            px2 = w_max
        if py2 > h_max:
            py1 -= (py2 - h_max)
            py2 = h_max

        # 4. 執行 1:1 採樣
        patch = full_image[py1:py2, px1:px2]
        
        # 5. 安全墊片：若原圖太小則 Resize (保底措施)
        if patch.shape[0] != 64 or patch.shape[1] != 64:
            patch = cv2.resize(patch, (64, 64))
            
        # 6. 轉灰階與歸一化
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
        patch_norm = patch.astype(np.float32) / 255.0
        return patch_norm[np.newaxis, np.newaxis, :, :]

    def predict_batch(self, patches_data):
        """
        利用 OpenVINO 進行真實 Batch 推論
        """
        if not patches_data:
            return []
            
        # 垂直堆疊各個 patch 形成一個 Batch N
        batch_input = np.vstack(patches_data)
        
        # 執行推論
        results = self.compiled_model([batch_input])[self.output_layer]
        
        processed_results = []
        for pred in results:
            cls_id = np.argmax(pred)
            # 簡易 Softmax 計算信心度
            exp_pred = np.exp(pred - np.max(pred))
            conf = exp_pred[cls_id] / np.sum(exp_pred)
            processed_results.append((cls_id, float(conf)))
            
        return processed_results

    def predict(self, full_image, roi):
        patch_data = self.preprocess_roi(full_image, roi)
        return self.predict_batch([patch_data])[0]

if __name__ == "__main__":
    # 單元測試：加載真實 ONNX 並測試一個 Fake ROI
    try:
        classifier = PCBDeepClassifier()
        dummy_img = np.zeros((640, 640), dtype=np.uint8)
        dummy_roi = [100, 100, 164, 164]
        
        clsid, conf = classifier.predict(dummy_img, dummy_roi)
        print(f"✅ Real AI Prediction: Class {clsid} ({classifier.classes[clsid]}), Conf: {conf:.4f}")
    except Exception as e:
        print(f"Check if python_research/pcb_classifier_v1.onnx exists. Error: {e}")
