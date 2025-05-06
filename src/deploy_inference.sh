#!/bin/bash
# deploy_inference.sh

echo "Starting U-Net TensorRT inference server..."
docker run --gpus all -v $(pwd):/workspace deepmedsynth_inference \
    python infer_unet_tensorrt.py
