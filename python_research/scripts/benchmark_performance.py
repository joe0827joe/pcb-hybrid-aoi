import os
import sys
import time
import cv2
import numpy as np

# 加入 src 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
# 現在腳本在 python_research/scripts，src 在 ../src
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))
sys.path.append(src_path)

from cv_prototype_v1 import PCBVisionPrototype
from dl_inference_v1 import PCBDeepClassifier
from hybrid_pipeline_v1 import HybridAOIPipeline

def run_benchmarks():
    print("========================================")
    print("   PCB-Hybrid-AOI Performance Benchmark ")
    print("========================================\n")

    pr_root = os.path.dirname(current_dir)
    fixture_dir = os.path.join(pr_root, "tests", "data", "fixtures")
    model_path = os.path.join(pr_root, "models", "pcb_classifier_v1.onnx")
    
    temp_path = os.path.join(fixture_dir, "sample_temp.jpg")
    test_path = os.path.join(fixture_dir, "sample_test.jpg")

    if not os.path.exists(temp_path) or not os.path.exists(model_path):
        print("❌ Error: Fixtures or Model not found. Run tests first to generate them.")
        return

    # 1. CV Stage Benchmark
    print("--- [Stage 1: CV Prototype] ---")
    detector = PCBVisionPrototype(debug=False)
    latencies = []
    for _ in range(10):
        _, metrics = detector.process(temp_path, test_path)
        latencies.append(metrics['total_latency_ms'] - metrics['load_time_ms'])
    avg_cv = sum(latencies) / len(latencies)
    print(f"Avg Logic Latency (10 runs): {avg_cv:.2f} ms (Target: < 50ms)")
    print("Status: " + ("✅ PASSED" if avg_cv < 50 else "❌ FAILED"))

    # 2. DL Stage Benchmark (Batch)
    print("\n--- [Stage 2: DL Inference Batch] ---")
    classifier = PCBDeepClassifier(model_path=model_path)
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    roi = [100, 100, 164, 164]
    patch = classifier.preprocess_roi(img, roi)
    patches = [patch] * 10
    
    start = time.time()
    classifier.predict_batch(patches)
    lat_batch = (time.time() - start) * 1000
    print(f"Batch (10 ROIs) Latency: {lat_batch:.2f} ms (Target: < 50ms)")
    print("Status: " + ("✅ PASSED" if lat_batch < 50 else "❌ FAILED"))

    # 3. End-to-End Pipeline
    print("\n--- [End-to-End: Hybrid Pipeline] ---")
    pipeline = HybridAOIPipeline(model_path=model_path)
    e2e_latencies = []
    for _ in range(10):
        _, latency = pipeline.inspect(temp_path, test_path)
        e2e_latencies.append(latency)
    avg_e2e = sum(e2e_latencies) / len(e2e_latencies)
    print(f"Avg E2E Latency (10 runs): {avg_e2e:.2f} ms (Target: < 130ms)")
    print("Status: " + ("✅ PASSED" if avg_e2e < 130 else "❌ FAILED"))

    # 寫入報告
    report_path = os.path.join(pr_root, "results", "performance_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Performance Benchmark Report\n")
        f.write(f"CV Stage Avg: {avg_cv:.2f} ms\n")
        f.write(f"DL Batch (10): {lat_batch:.2f} ms\n")
        f.write(f"End-to-End Avg: {avg_e2e:.2f} ms\n")
        f.write(f"Status: {'SUCCESS' if avg_e2e < 130 else 'PERF_DEGRADED'}\n")

if __name__ == "__main__":
    run_benchmarks()
