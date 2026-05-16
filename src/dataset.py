import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(7),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)), 
    transforms.ToTensor(), 
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def get_dataloaders(data_dir, batch_size=32, num_workers=2, pin_memory=True):
    base_dataset = datasets.ImageFolder(data_dir)
    class_names = base_dataset.classes

    indices = np.arange(len(base_dataset))
    labels = np.array(base_dataset.targets)

    train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_labels, random_state=42)

    train_dataset_full = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(data_dir, transform=val_test_transform)
    test_dataset_full = datasets.ImageFolder(data_dir, transform=val_test_transform)

    train_set = Subset(train_dataset_full, train_idx)
    val_set = Subset(val_dataset_full, val_idx)
    test_set = Subset(test_dataset_full, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, class_names
