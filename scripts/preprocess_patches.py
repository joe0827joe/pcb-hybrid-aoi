import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_patches():
    BASE_DIR = "data/DeepPCB-master/PCBData"
    OUTPUT_DIR = "data/patches"
    PATCH_SIZE = 64
    
    for i in range(7):
        os.makedirs(os.path.join(OUTPUT_DIR, str(i)), exist_ok=True)

    print("--- Starting Robust Patch Extraction (Decoupled Folders) ---")
    
    patch_count = 0
    # 遍歷所有 groupXXXXX 目錄
    groups = [d for d in os.listdir(BASE_DIR) if d.startswith("group")]
    
    for group in tqdm(groups):
        group_path = os.path.join(BASE_DIR, group)
        
        # 識別兩個關鍵目錄: 存圖的 XXXXX 與 存標註的 XXXXX_not
        img_sub = group.replace("group", "")
        ann_sub = img_sub + "_not"
        
        img_dir = os.path.join(group_path, img_sub)
        ann_dir = os.path.join(group_path, ann_sub)
        
        if not (os.path.exists(img_dir) and os.path.exists(ann_dir)):
            continue
            
        test_images = [f for f in os.listdir(img_dir) if f.endswith("_test.jpg")]
        
        for img_name in test_images:
            img_path = os.path.join(img_dir, img_name)
            # 對接標註檔: 00041000_test.jpg -> 00041000.txt
            ann_name = img_name.replace("_test.jpg", ".txt")
            ann_path = os.path.join(ann_dir, ann_name)
            
            if not os.path.exists(ann_path): continue
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            h_max, w_max = img.shape
            
            with open(ann_path, 'r') as f:
                for line in f:
                    line_data = line.strip().replace(',', ' ')
                    parts = line_data.split()
                    if len(parts) < 5: continue
                    
                    try:
                        x1, y1, x2, y2, clsid = map(int, parts[:5])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        px1, py1 = max(0, cx - PATCH_SIZE // 2), max(0, cy - PATCH_SIZE // 2)
                        px2, py2 = min(w_max, px1 + PATCH_SIZE), min(h_max, py1 + PATCH_SIZE)
                        
                        patch = img[py1:py2, px1:px2]
                        if patch.size == 0: continue
                        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                            patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                        
                        out_name = f"{img_name}_{x1}_{y1}.jpg"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, str(clsid), out_name), patch)
                        patch_count += 1
                    except Exception:
                        continue
            
            # --- 新增: 隨機抓取 5 個背景補丁 (不與標註重疊的區域) ---
            # 從模板圖 (temp) 取樣以確保是乾淨背景
            temp_path = img_path.replace("_test.jpg", "_temp.jpg")
            img_temp = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            if img_temp is not None:
                for bidx in range(5):
                    bx = np.random.randint(0, w_max - PATCH_SIZE)
                    by = np.random.randint(0, h_max - PATCH_SIZE)
                    bg_patch = img_temp[by:by+PATCH_SIZE, bx:bx+PATCH_SIZE]
                    cv2.imwrite(os.path.join(OUTPUT_DIR, "0", f"bg_{img_name}_{bidx}.jpg"), bg_patch)
                    patch_count += 1

    print(f"✅ Success: Extracted {patch_count} patches to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_patches()
