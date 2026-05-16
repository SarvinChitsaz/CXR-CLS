import os
import torch
from src.utils import get_device
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import train_model
from src.eval import evaluate_model
from src.visualize import show_visualization
from src.gradcam import GradCAM, show_gradcam

device = get_device()
print("Device:", device)

data_dir = os.path.expanduser("~/Desktop/X-Ray/Chest X-Ray Covid19 Pneumonia")
train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir)

model = get_model(num_classes=len(class_names)).to(device)

model, train_losses, val_losses = train_model(model, train_loader, val_loader, device, epochs=10)

evaluate_model(model, test_loader, device, class_names)

show_visualization(model, test_loader, class_names, device, n=3)

target_layer = model.layer4[1].conv2
gradcam = GradCAM(model, target_layer)
show_gradcam(model, test_loader, class_names, device, gradcam, n=5)

os.makedirs("checkpoints", exist_ok=True)
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": None,
    "class_names": class_names
}
torch.save(checkpoint, "checkpoints/xray_resnet18_checkpoint.ckpt")
print("Checkpoint saved.")
