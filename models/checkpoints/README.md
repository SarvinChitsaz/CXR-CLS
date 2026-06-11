# 💾 Model Checkpoint (Pretrained Weights)

This repository uses a pretrained ResNet18 model for chest X-ray classification (COVID-19, NORMAL, PNEUMONIA).

The trained checkpoint is not included in the repository due to file size limitations.

# 📥 Download Checkpoint

Download the pretrained model from Google Drive:

https://drive.google.com/file/d/1k-sHICymGiRqQxUOAPkEP8UW41Sz689z/view?usp=share_link

After downloading, you will get:

xray_resnet18_checkpoint.ckpt

# 📁 File Placement

Place the downloaded file in the following path:

CXR-CLS/checkpoints/xray_resnet18_checkpoint.ckpt

# Checkpoint Information

The checkpoint includes:

- Trained ResNet18 weights
- Optimizer state
- Class labels

Classes:
- COVID-19
- NORMAL
- PNEUMONIA

# How to Use

Load the checkpoint using the same ResNet18 architecture used during training and run inference directly.

Make sure:
- Input size is 224×224
- Images are converted to RGB (3 channels)
- ImageNet normalization is applied

# ⚠️ Notes

- This checkpoint is required for inference
- Without it, the model must be retrained from scratch
