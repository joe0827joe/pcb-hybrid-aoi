# PCB-AOI Detection Schema (V1.1)

> **Owner:** `architect_01`
> **Scope:** Python/C++ Cross-Language Data Protocol (High Performance)

---

## 1. Class Mapping (0-6)
To ensure alignment between Python prototyping and C++ deployment, all model outputs MUST use the following ID mapping:

| ID | Class Name | Description | Confidence Threshold (Min) |
| :--- | :--- | :--- | :--- |
| **0** | **Background** | Normal trace or plate (Non-defect) | N/A |
| **1** | **Open** | Trace discontinuity | 0.85 (Critical) |
| **2** | **Short** | Unintended connection | 0.85 (Critical) |
| **3** | **Mousebite** | Partial trace reduction | 0.75 |
| **4** | **Spur** | Protrusion from trace | 0.70 |
| **5** | **Copper** | Isolated copper residue | 0.65 |
| **6** | **Pin-hole** | Small hole in trace | 0.60 |

---

## 2. Prediction Object Format & Memory Alignment
In order to minimize **Cache Misses** and **Memory Padding overhead** in C++17 production environments, the Prediction Struct should be ordered as follows (assuming 4-byte/8-byte alignment):

### C++ Struct Recommendation (`alignas(16)`):
```cpp
struct alignas(16) DetectionResult {
    float confidence;      // 4 bytes - (Score 0.0 ~ 1.0)
    int32_t x1, y1;        // 8 bytes - ROI T-L Coordinate
    int32_t x2, y2;        // 8 bytes - ROI B-R Coordinate
    uint8_t class_id;      // 1 byte  - (See Class Mapping)
    uint8_t reserved[3];   // 3 bytes - Padding to satisfy 4-byte alignment
}; // Total: 24 bytes
```

### Python/C++ JSON Serialized Format:
```json
{
  "class_id": 1,
  "confidence": 0.92,
  "roi": [x1, y1, x2, y2]
}
```

---

## 3. Preprocessing & Input Handling (64x64)
To avoid interpolation artifacts that blur critical trace features (Open vs. Mousebite):

- **Image Normalization**: 0.0 ~ 1.0 (float32 at inference, 8-bit at storage).
- **Scale Strategy**: **Center Padding (Padding-to-Center)**.
    - **Step 1**: Calculate aspect ratio of original ROI.
    - **Step 2**: Letterbox/Padding to maintain 1:1 ratio with `zero-padding (0x00 / Black)`.
    - **Step 3**: Resize to 64x64 using **Lanczos interpolation** (avoid Bilinear if possible to maintain edge sharpness).

---

## 4. Deployment Requirements
- **Inference Engine**: OpenVINO 2022.3+ / ONNX Runtime.
- **Precision**: INT8 Quantization (preferred) or FP32.
- **Fail-soft Logic**: Any ROI with `confidence < ClassThreshold` MUST be flagged as `LOW_CONFIDENCE` and passed to a secondary heuristic CV validator.
