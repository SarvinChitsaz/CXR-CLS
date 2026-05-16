```
dataset/
│
├── COVID/
├── NORMAL/
└── PNEUMONIA/
```

# Dataset for CXR-CLS

This folder contains the chest X-ray images used for training, validation, and testing in the CXR-CLS pipeline.

## Expected Structure

The dataset should be organized into class-specific folders as shown below:

- `COVID/` → Contains chest X-ray images of COVID-19 cases.
- `NORMAL/` → Contains chest X-ray images of healthy patients.
- `PNEUMONIA/` → Contains chest X-ray images of non-COVID pneumonia cases.

Each folder should include all images corresponding to that class. Supported image formats are `.png` and `.jpg`.

## Dataset Guidelines

- Make sure all images are correctly labeled in their respective folders.
- Images can be of varying resolutions; the pipeline will automatically resize them to 224x224 during preprocessing.
- Grayscale images are converted to 3-channel images for compatibility with the ResNet18 model.
- Avoid corrupted or unreadable image files, as they may cause errors during training or evaluation.

## Usage

The CXR-CLS pipeline automatically performs:

- Splitting the dataset into training, validation, and test sets using stratified sampling.
- Dynamic preprocessing including resizing, normalization, and augmentation.
- Loading images into PyTorch DataLoaders for model training and evaluation.

## Notes

- Place your dataset inside this folder before running the pipeline.
- Ensure folder names match exactly (`COVID`, `NORMAL`, `PNEUMONIA`) for proper data loading.
- Additional images for augmentation or experiments can be added, but maintain the same folder structure.
