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
        修正裁切邏輯：將 ROI 採集後 Resize 為模型所需的 64x64
        """
        x1, y1, x2, y2 = map(int, roi)
        h_max, w_max = full_image.shape[:2]
        
        # 確保邊界不溢出
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_max, x2), min(h_max, y2)
        
        patch = full_image[y1:y2, x1:x2]
        if patch.size == 0:
            return np.zeros((1, 1, 64, 64), dtype=np.float32)
            
        patch_resized = cv2.resize(patch, (64, 64))
        # 轉灰階 (若原圖是 BGR)
        if len(patch_resized.shape) == 3:
            patch_resized = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
            
        # 歸一化與維度對整 (N, C, H, W)
        patch_norm = patch_resized.astype(np.float32) / 255.0
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
