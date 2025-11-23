from huggingface_hub import hf_hub_download
import shutil
import os

print("⬇️  Downloading YOLOv8 License Plate Model...")

# This downloads a fine-tuned YOLOv8 model specifically for license plates
# Source: https://huggingface.co/yasirfaizahmed/license-plate-object-detection
model_path = hf_hub_download(
    repo_id="yasirfaizahmed/license-plate-object-detection",
    filename="best.pt"
)

# Move/Rename it to our local folder for easy access
destination = "license_plate_detector.pt"
shutil.copy(model_path, destination)

print(f"✅ Model downloaded successfully as '{destination}'")
print("🚀 You can now run 'python app.py'")