import matplotlib.pyplot as plt
import numpy as np
import random

def show_visualization(model, loader, class_names, device, n=3):
    model.eval()
    correct_imgs = []
    wrong_imgs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            for i in range(len(x)):
                img = x[i].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])

                img = img * std + mean
                img = np.clip(img, 0, 1)

                true_label = class_names[y[i]]
                pred_label = class_names[preds[i]]

                if preds[i] == y[i]:
                    correct_imgs.append((img, true_label, pred_label))
                else:
                    wrong_imgs.append((img, true_label, pred_label))

    correct_imgs = random.sample(correct_imgs, min(n, len(correct_imgs)))
    wrong_imgs = random.sample(wrong_imgs, min(n, len(wrong_imgs)))

    plt.figure(figsize=(10, 4))
    plt.suptitle("Correct Predictions")
    for i, (img, true, pred) in enumerate(correct_imgs):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(f"T: {true}\nP: {pred}")
        plt.axis("off")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.suptitle("Wrong Predictions")
    for i, (img, true, pred) in enumerate(wrong_imgs):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(f"T: {true}\nP: {pred}")
        plt.axis("off")
    plt.show()
