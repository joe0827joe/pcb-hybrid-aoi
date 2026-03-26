import os
import subprocess
import sys

def setup_data():
    """
    PCB-Hybrid-AOI: Dataset Provisioning Script
    Objective: Automate DeepPCB dataset download and placement.
    """
    # Define project structure
    # Script is in python_research/scripts/, so root is 2 levels up
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", "DeepPCB")
    alt_data_dir = os.path.join(base_dir, "data", "DeepPCB-master")
    
    print("-" * 40)
    print("🚀 PCB-AOI: Dataset Setup Protocol")
    print("-" * 40)

    if os.path.exists(data_dir):
        print(f"✅ Dataset already exists at: {data_dir}")
        return
    
    if os.path.exists(alt_data_dir):
        print(f"✅ Found existing dataset at: {alt_data_dir}")
        return

    print(f"📦 Source: https://github.com/tangy7/DeepPCB.git")
    print(f"📂 Target: {data_dir}")
    
    # Ensure data parent directory exists
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)

    try:
        # Clone with depth=1 to save disk space and bandwidth
        subprocess.run([
            "git", "clone", "--depth", "1", 
            "https://github.com/tangy7/DeepPCB.git", data_dir
        ], check=True)
        print("✨ DeepPCB dataset is ready!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_data()
