import subprocess
import os

def debug_git_commit():
    """
    PCB-Hybrid-AOI: Git Commit Debugger
    Aims to trap why the agent's commit command is failing.
    """
    print("-" * 40)
    print("🔍 Probing Git Commit Status...")
    print("-" * 40)
    
    try:
        # 1. 確保 add 成功
        subprocess.run(["git", "add", "README.md", "scripts/setup_data.py"], check=True)
        print("✅ Git Add Finished.")
        
        # 2. 執行 Commit 並捕捉所有輸出 (包含成功或失敗)
        res = subprocess.run(
            ["git", "commit", "-m", "chore(data): automate dataset provisioning for DeepPCB"],
            capture_output=True, text=True
        )
        
        if res.returncode == 0:
            print("🚀 STATUS: COMMIT SUCCESS!")
            print(res.stdout)
        else:
            print("🔴 STATUS: COMMIT FAILED.")
            print("--- STDOUT ---")
            print(res.stdout)
            print("--- STDERR ---")
            print(res.stderr)
            
    except Exception as e:
        print(f"❌ Python Level Exception: {e}")

if __name__ == "__main__":
    debug_git_commit()
