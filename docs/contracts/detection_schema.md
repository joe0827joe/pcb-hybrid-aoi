# PCB-AOI Detection Schema (V1.0)

> **Owner:** `architect_01`
> **Scope:** Python/C++ Data Protocol

## 1. Class Mapping (0-6)
為了開發端與部署端統一，所有模型輸出必須對齊以下類別集：

| ID | Class Name | Description |
| :--- | :--- | :--- |
| **0** | **Background** | Normal trace or plate (Non-defect) |
| **1** | **Open** | Trace discontinuity |
| **2** | **Short** | Unintended connection between traces |
| **3** | **Mousebite** | Partial trace reduction |
| **4** | **Spur** | Small protrusion from a trace |
| **5** | **Copper** | Isolated copper residue |
| **6** | **Pin-hole** | Small hole in a trace |

## 2. ROI Coordination Format
*   **Format**: `[x1, y1, x2, y2]` (Top-left to Bottom-right)
*   **Coordinate System**: Pixel-based, Origin at Top-Left.
*   **Scaling**: Coordinates must be relative to the 1:1 original image resolution.

## 3. Deployment Requirements
*   **Inference Engine**: OpenVINO 2022+ / ONNX Runtime.
*   **Input Size**: 64x64 pixels (Grayscale).
*   **Color Space**: Luma (Y) / 8-bit.
