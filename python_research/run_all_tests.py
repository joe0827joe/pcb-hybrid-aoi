import unittest
import os
import sys

def run_tests():
    # 加入當前目錄到路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    sys.path.append(os.path.join(current_dir, ".."))

    loader = unittest.TestLoader()
    # 查找所有 test_*.py 檔案
    suite = loader.discover(start_dir=current_dir, pattern='test_*.py')

    print("========================================")
    print("   PCB-Hybrid-AOI Test Suite Runner     ")
    print("========================================\n")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 寫入簡要測試結果到 test_data 供效能稽核使用
    report_path = os.path.join(current_dir, "test_data", "unit_test_summary.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Unit & Integration Test Summary\n")
        f.write(f"Tests Run: {result.testsRun}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Failures: {len(result.failures)}\n")
        f.write(f"Success: {result.wasSuccessful()}\n")
        
    if not result.wasSuccessful():
        print("\n❌ One or more tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
