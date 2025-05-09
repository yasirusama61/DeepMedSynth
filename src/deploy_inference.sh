#!/bin/bash
# deploy_inference.sh
# Description: Launch DeepMedSynth inference inside a Docker container using TensorRT or OpenVINO.

set -e  # exit on error

# === Config ===
MODEL_DIR="$(pwd)/segmentation_results_v3"
SCRIPT_NAME="infer_unet_tensorrt.py"  # Change to infer_unet_openvino.py for OpenVINO
CONTAINER_NAME="deepmedsynth_inference"
DEVICE_TYPE="gpu"  # Options: gpu, cpu

# === Check Docker installed ===
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed. Please install Docker first."
    exit 1
fi

# === Check model and script existence ===
if [ ! -f "$MODEL_DIR/$SCRIPT_NAME" ]; then
    echo "❌ Inference script not found: $MODEL_DIR/$SCRIPT_NAME"
    exit 1
fi

# === Run Docker Inference ===
echo "🚀 Launching $CONTAINER_NAME with $SCRIPT_NAME on $DEVICE_TYPE..."

docker run --rm \
    --name $CONTAINER_NAME \
    -v "$MODEL_DIR":/workspace \
    -w /workspace \
    --gpus all \
    deepmedsynth_inference \
    python $SCRIPT_NAME

# === Done ===
echo "✅ Inference complete using $SCRIPT_NAME in $CONTAINER_NAME container."