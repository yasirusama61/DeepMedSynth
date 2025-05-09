# 📊 Inference Benchmark Summary (DeepMedSynth v3)

This document summarizes real-world inference performance of the DeepMedSynth U-Net model on edge devices using both TensorRT and OpenVINO runtimes.

📁 [Back to Deployment Guide](deployment_on_edge.md)

---

## 🏎️ TensorRT (Jetson Xavier NX, FP16)
- **Model Format:** TensorRT Engine (FP16)
- **Input Resolution:** 128×128×3
- **Batch Size:** 1
- **Average Latency:** 6.42 ms
- **Throughput:** ~155 FPS
- **Peak GPU Memory:** 382 MB
- **Device Temperature:** ~54°C under load
- **Power Draw:** ~8.7W

---

## 🧠 OpenVINO (Intel Core i5-1145G7, CPU-only)
- **Model Format:** IR (XML + BIN)
- **Average Latency:** 19.87 ms
- **Throughput:** ~50 FPS
- **Peak RAM Usage:** 410 MB
- **CPU Utilization:** 68%
- **Precision Mode:** FP32 (default)

---

## 📦 Model Footprint
- **ONNX Export:** 18.2 MB
- **TensorRT Engine (FP16):** 9.4 MB
- **OpenVINO IR:** 17.5 MB (XML + BIN)

---

## 🔬 FLOPs and Parameters
- **Model Type:** 2D U-Net (3 input channels, 1 output channel)
- **Estimated FLOPs:** 8.2 GFLOPs
- **Total Parameters:** ~2.4M

---

## ✅ Deployment Readiness Summary
| Platform        | Latency (ms) | FPS   | Memory   | Precision |
|----------------|--------------|--------|----------|-----------|
| Jetson (TRT)   | 6.42         | 155 FPS| 382 MB   | FP16      |
| Intel (OV)     | 19.87        | 50 FPS | 410 MB   | FP32      |


✅ Both platforms support batch inference, low memory, and sub-20ms latency — ideal for real-time medical AI at the edge.

---

