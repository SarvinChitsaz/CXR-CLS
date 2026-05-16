import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        loss = output[:, target_class]
        loss.backward()

        gradients = self.gradients[0].detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (224, 224))
        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def show_gradcam(model, loader, class_names, device, gradcam, n=5):
    model.eval()
    correct_samples = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        for i in range(len(x)):
            if preds[i] == y[i]:
                correct_samples.append((x[i], y[i], preds[i]))

    random_samples = random.sample(correct_samples, min(n, len(correct_samples)))

    for sample in random_samples:
        x_img, y_img, pred_img = sample
        input_tensor = x_img.unsqueeze(0)

        cam = gradcam.generate_cam(input_tensor, target_class=pred_img.item())

        img = x_img.detach().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        overlay = heatmap * 0.4 + img
        overlay = overlay / overlay.max()

        true_label = class_names[y_img]
        pred_label = class_names[pred_img]

        plt.figure(figsize=(5, 2.5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Original\nTrue: {true_label}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f"Predicted: {pred_label}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    gradcam.remove_hooks()
