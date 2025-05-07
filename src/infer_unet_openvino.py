# infer_unet_openvino.py

import os
import cv2
import numpy as np
from openvino.runtime import Core
from pathlib import Path
from time import time

# === Config ===
IR_MODEL_DIR = "segmentation_results_v3/openvino_ir"
IMAGE_DIR = "test_images"
OUTPUT_DIR = "openvino_predictions"
DEVICE = "CPU"  # or "GPU" if OpenVINO iGPU is available

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load OpenVINO Model ===
print("🔄 Loading OpenVINO IR model...")
ie = Core()
model = ie.read_model(model=Path(IR_MODEL_DIR) / "unet_model.xml")
compiled_model = ie.compile_model(model=model, device_name=DEVICE)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
print("✅ Model loaded and compiled")

# === Preprocessing Function ===
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch
    return img

# === Postprocessing ===
def postprocess(output):
    mask = output.squeeze()  # [1,1,128,128] -> [128,128]
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

# === Inference Loop ===
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".png")])
latencies = []

print("🚀 Running inference...")
for file in image_files:
    img_path = os.path.join(IMAGE_DIR, file)
    img_input = preprocess(img_path)

    start = time()
    result = compiled_model([img_input])[output_layer]
    latency = (time() - start) * 1000  # ms
    latencies.append(latency)

    mask = postprocess(result)
    save_path = os.path.join(OUTPUT_DIR, f"mask_{file}")
    cv2.imwrite(save_path, mask)

    print(f"🖼️ {file} → done ({latency:.2f} ms)")

print(f"\n⏱️ Average inference time: {np.mean(latencies):.2f} ms")
print(f"📂 Saved all masks to: {OUTPUT_DIR}")
