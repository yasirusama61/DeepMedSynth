# Inference Benchmark Summary (DeepMedSynth v3)

## TensorRT (Jetson Xavier NX - FP16)
- Input: 128×128
- Average latency: 6.42 ms
- FPS: ~155
- Max GPU Memory: 380 MB

## OpenVINO (Intel i5 CPU)
- Input: 128×128
- Average latency: 19.87 ms
- FPS: ~50
- CPU Utilization: 65%
- RAM: 410 MB

## Model Size
- ONNX: 18.2 MB
- TensorRT Engine: 9.4 MB (FP16)
- OpenVINO IR: 17.5 MB

## FLOPs (estimated)
- U-Net: ~8.2 GFLOPs
