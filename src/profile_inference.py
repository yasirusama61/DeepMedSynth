# scripts/profile_inference.py

import torch
import time
import psutil
import os
from models import build_unet

MODEL_PATH = "segmentation_results_v3/best_model.pth"

# Setup
model = build_unet(input_shape=(128, 128, 3))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

input_tensor = torch.randn(1, 3, 128, 128)

# Measure latency
with torch.no_grad():
    times = []
    for _ in range(50):
        start = time.time()
        _ = model(input_tensor)
        end = time.time()
        times.append(end - start)

print(f"✅ Avg Inference Time: {1000 * sum(times)/len(times):.2f} ms")

# Measure memory
process = psutil.Process(os.getpid())
mem_usage = process.memory_info().rss / (1024 ** 2)
print(f"💾 Peak RAM usage: {mem_usage:.2f} MB")
