import shutil
import os

src_temp = r"data/DeepPCB-master/PCBData/group00041/00041/00041000_temp.jpg"
src_test = r"data/DeepPCB-master/PCBData/group00041/00041/00041000_test.jpg"
dst_dir = r"python_research/test_data/fixtures"

os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src_temp, os.path.join(dst_dir, "sample_temp.jpg"))
shutil.copy(src_test, os.path.join(dst_dir, "sample_test.jpg"))
print("Successfully copied test fixtures.")
