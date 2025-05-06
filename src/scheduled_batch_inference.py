# scheduled_batch_inference.py

import os
import time
import numpy as np
from preprocess_hospital_data import preprocess_file
from infer_unet_tensorrt import run_inference
from postprocess_masks import postprocess_output

RAW_DIR = "/hospital_data/raw_scans/"
PREPROCESSED_DIR = "/hospital_data/preprocessed_data/"
PREDICTIONS_DIR = "/hospital_data/predictions/"
FINAL_DIR = "/hospital_data/final_reports/"

def process_new_files():
    raw_files = {f for f in os.listdir(RAW_DIR) if f.endswith(".nii") or f.endswith(".nii.gz")}
    processed_files = {f.replace(".npy", ".nii") for f in os.listdir(PREPROCESSED_DIR)}

    new_files = raw_files - processed_files
    for file in new_files:
        print(f"🚀 Processing new file: {file}")
        preprocess_file(os.path.join(RAW_DIR, file))
        preproc_path = os.path.join(PREPROCESSED_DIR, file.replace(".nii", ".npy").replace(".nii.gz", ".npy"))
        pred_mask = run_inference(preproc_path)
        postprocess_output(pred_mask, os.path.join(FINAL_DIR, file.replace(".nii", "_mask.png")))
        print(f"✅ Finished {file}")

if __name__ == "__main__":
    while True:
        process_new_files()
        print("⏰ Waiting for next check...")
        time.sleep(600)  # every 10 minutes
