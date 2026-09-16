import torch
import matplotlib.pyplot as plt
from src.config import Config
from src.kaggle_data import build_transforms
import cv2
import numpy as np
import pydicom

def get_sample_image():
    # Load a sample dicom from the train set
    dcm_path = "data_subset/train_images/1.2.826.0.1.3680043.8.498.10983416091338854252058678062116243962/1.2.826.0.1.3680043.8.498.89224830765395350281741857307394414871/1.2.826.0.1.3680043.8.498.10682585856062880070971310957835638718.dcm"
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array
    
    # Simple normalization just for visualization
    img = img.astype(np.float32)
    if img.max() > 0:
        img /= img.max()
    img = (img * 255).astype(np.uint8)
    
    # Convert 1-channel to 3-channel
    return np.stack([img]*3, axis=-1)

def main():
    cfg = Config()
    # Ensure augmentations are enabled
    cfg.aug_enabled = True
    
    # Build the new augmentation pipeline
    transform = build_transforms(cfg, is_train=True, in_channels=3)
    
    img = get_sample_image()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Original
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    # 4 Random Augmentations
    for i in range(1, 5):
        augmented = transform(image=img)["image"]
        # Denormalize
        mean = torch.tensor([0.485, 0.485, 0.485]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.229, 0.229]).view(3, 1, 1)
        augmented = augmented * std + mean
        
        # CHW to HWC
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = np.clip(augmented, 0, 1)
        
        axes[i].imshow(augmented)
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis("off")
        
    plt.tight_layout()
    plt.savefig("augmentation_preview.png")
    print("Saved preview to augmentation_preview.png")

if __name__ == "__main__":
    main()
