import torch
import torch.onnx
import sys
import os

# 動態加入路徑以尋找模型定義
sys.path.append(os.path.abspath("python_research"))
from train_classifier import TinyPCBNet # 引用模型結構

def export_to_onnx():
    INPUT_PATH = "python_research/pcb_classifier_v1.pth"
    OUTPUT_PATH = "python_research/pcb_classifier_v1.onnx"
    
    # 初始化模型並載入權重
    model = TinyPCBNet()
    model.load_state_dict(torch.load(INPUT_PATH))
    model.eval()

    # 模擬輸入 Tensor (Batch=1, Channels=1, 64x64)
    dummy_input = torch.randn(1, 1, 64, 64)
    
    # 導出為 ONNX，並開啟動態 Batch Size
    torch.onnx.export(model, dummy_input, OUTPUT_PATH, 
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={'input' : {0 : 'batch_size'}, 
                                   'output' : {0 : 'batch_size'}},
                      opset_version=11)
    print(f"✅ Model successfully exported to {OUTPUT_PATH}")

if __name__ == "__main__":
    export_to_onnx()
