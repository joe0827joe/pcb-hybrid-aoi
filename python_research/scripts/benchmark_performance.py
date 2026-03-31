import os
import sys
import time
import cv2
import numpy as np
import json
import argparse

# 加入 src 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
# 現在腳本在 python_research/scripts，src 在 ../src
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))
sys.path.append(src_path)

from cv_prototype_v1 import PCBVisionPrototype
from dl_inference_v1 import PCBDeepClassifier
from hybrid_pipeline_v1 import HybridAOIPipeline

def run_benchmarks(threshold_ms=130):
    print("========================================")
    print("   PCB-Hybrid-AOI Performance Benchmark ")
    print(f"   (Threshold: {threshold_ms}ms) ")
    print("========================================\n")

    pr_root = os.path.dirname(current_dir)
    results_dir = os.path.join(pr_root, "results")
    fixture_dir = os.path.join(pr_root, "tests", "data", "fixtures")
    model_path = os.path.join(pr_root, "models", "pcb_classifier_v1.onnx")
    
    temp_path = os.path.join(fixture_dir, "sample_temp.jpg")
    test_path = os.path.join(fixture_dir, "sample_test.jpg")

    if not os.path.exists(temp_path) or not os.path.exists(model_path):
        print("❌ Error: Fixtures or Model not found. Run tests first to generate them.")
        sys.exit(1)

    # 1. CV Stage Benchmark
    print("--- [Stage 1: CV Prototype] ---")
    detector = PCBVisionPrototype(debug=False)
    latencies = []
    # Warmup
    detector.process(temp_path, test_path)
    
    for _ in range(10):
        _, metrics = detector.process(temp_path, test_path)
        latencies.append(metrics['total_latency_ms'])
    avg_cv = sum(latencies) / len(latencies)
    print(f"Avg CV Latency (10 runs): {avg_cv:.2f} ms (Target: < 50ms)")
    print("Status: " + ("✅ PASSED" if avg_cv < 50 else "❌ FAILED"))

    # 2. DL Stage Benchmark (Batch)
    print("\n--- [Stage 2: DL Inference Batch] ---")
    classifier = PCBDeepClassifier(model_path=model_path)
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    roi = [100, 100, 164, 164]
    patch = classifier.preprocess_roi(img, roi)
    patches = [patch] * 10
    
    # Warmup
    classifier.predict_batch(patches)
    
    start = time.time()
    for _ in range(5):
        classifier.predict_batch(patches)
    lat_batch = ((time.time() - start) / 5) * 1000
    print(f"Batch (10 ROIs) Latency: {lat_batch:.2f} ms (Target: < 50ms)")
    print("Status: " + ("✅ PASSED" if lat_batch < 50 else "❌ FAILED"))

    # 3. End-to-End Pipeline
    print("\n--- [End-to-End: Hybrid Pipeline] ---")
    pipeline = HybridAOIPipeline(model_path=model_path)
    # Warmup
    pipeline.inspect(temp_path, test_path)
    
    e2e_latencies = []
    for _ in range(10):
        _, latency = pipeline.inspect(temp_path, test_path)
        e2e_latencies.append(latency)
    avg_e2e = sum(e2e_latencies) / len(e2e_latencies)
    print(f"Avg E2E Latency (10 runs): {avg_e2e:.2f} ms (Target: < {threshold_ms}ms)")
    
    passed = avg_e2e < threshold_ms
    print("Status: " + ("✅ PASSED" if passed else "❌ FAILED"))

    # 產出報告資料
    report_data = {
        "version": "1.1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold_ms": threshold_ms,
        "metrics": {
            "cv_avg_ms": avg_cv,
            "dl_batch_10_ms": lat_batch,
            "e2e_avg_ms": avg_e2e
        },
        "status": "PASSED" if passed else "FAILED"
    }

    # 寫入 JSON 報告
    os.makedirs(results_dir, exist_ok=True)
    json_report_path = os.path.join(results_dir, "benchmark_report.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    
    # 保留文字報告供人類閱讀
    txt_report_path = os.path.join(results_dir, "performance_report.txt")
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("# Performance Benchmark Report\n")
        f.write(f"CV Stage Avg: {avg_cv:.2f} ms\n")
        f.write(f"DL Batch (10): {lat_batch:.2f} ms\n")
        f.write(f"End-to-End Avg: {avg_e2e:.2f} ms\n")
        f.write(f"Threshold: {threshold_ms} ms\n")
        f.write(f"Status: {'SUCCESS' if passed else 'PERF_DEGRADED'}\n")

    print(f"\n[OK] Reports saved to {results_dir}")
    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Benchmark for PCB-Hybrid-AOI")
    parser.add_argument("--threshold", type=str, default="130ms", help="Performance threshold (e.g., 130ms)")
    args = parser.parse_args()
    
    # 移除 ms 單位以便轉換為 int
    try:
        thr_val = int(args.threshold.replace("ms", ""))
    except ValueError:
        thr_val = 130
        
    run_benchmarks(threshold_ms=thr_val)
