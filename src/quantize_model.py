# scripts/quantize_model.py

import torch
import torchvision.models as models
from torch.quantization import quantize_dynamic
from models import build_unet

MODEL_PATH = "segmentation_results_v3/best_model.pth"
QUANTIZED_PATH = "segmentation_results_v3/unet_model_quantized.pth"

# === Load model
def load_model():
    model = build_unet(input_shape=(128, 128, 3))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# === Apply dynamic quantization (linear layers only)
def quantize_model(model):
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

if __name__ == "__main__":
    model = load_model()
    q_model = quantize_model(model)
    torch.save(q_model.state_dict(), QUANTIZED_PATH)
    print(f"✅ Quantized model saved at {QUANTIZED_PATH}")