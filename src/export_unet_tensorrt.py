# export_unet_tensorrt.py

# export_unet_tensorrt.py

"""
This script exports a trained U-Net model from PyTorch to ONNX,
then builds and saves a TensorRT engine for accelerated inference.

- Requires: torch, onnx, tensorrt, onnx-graphsurgeon
- Input shape: (1, 3, 128, 128)
"""

import torch
import onnx
from torch.onnx import export
import tensorrt as trt
from models import build_unet  # adjust import if needed

# === Load trained PyTorch model ===
model = build_unet(input_shape=(128,128,3))
model.load_state_dict(torch.load("segmentation_results_v3/best_model.pth", map_location="cpu"))
model.eval()

# === Export to ONNX ===
dummy_input = torch.randn(1, 3, 128, 128)
onnx_file = "unet_model.onnx"

export(model, dummy_input, onnx_file, input_names=["input"], output_names=["output"], opset_version=11)
print(f"✅ Exported PyTorch model to {onnx_file}")

# === Build TensorRT Engine ===
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_file, 'rb') as f:
    if not parser.parse(f.read()):
        print("❌ Failed to parse ONNX model:")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
    else:
        print(f"✅ Successfully parsed {onnx_file} for TensorRT")

config = builder.create_builder_config()
config.max_workspace_size = 1 << 28  # 256 MiB
engine = builder.build_engine(network, config)

# === Save Engine ===
engine_file = "unet_model.trt"
with open(engine_file, "wb") as f:
    f.write(engine.serialize())
print(f"✅ TensorRT engine saved as {engine_file}")
