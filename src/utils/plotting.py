# -- IMPORTS --
import os
import matplotlib
from monai.apps import download_and_extract
import matplotlib.pyplot as plt
import numpy as np
import torch

# -- CODE --
def plot_loss_metrics(train_losses, val_metrics, save_path="loss_metrics.png"):
    plt.figure(figsize=(12, 5), facecolor='white')
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color='skyblue', linewidth=2)
    plt.title('Training Loss per Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Dice)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_metrics, color='lightcoral', linewidth=2)
    plt.title('Validation Dice Score per Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_segmentation_predictions(image, ground_truth, prediction, save_path="prediction_sample.png", slice_idx=None):
    if slice_idx is None:
        slice_idx = image.shape[-1] // 2 

    image_slice = image[0, :, :, slice_idx] if image.ndim == 4 else image[:, :, slice_idx]
    gt_slice = ground_truth[0, :, :, slice_idx] if ground_truth.ndim == 4 else ground_truth[:, :, slice_idx]
    pred_slice = prediction[0, :, :, slice_idx] if prediction.ndim == 4 else prediction[:, :, slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='white')

    # Original Image (HU Scaled)
    axes[0].imshow(image_slice.cpu().numpy().T, cmap='gray', origin='lower')
    axes[0].set_title('Original Image (HU Scaled)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Ground Truth Overlay
    axes[1].imshow(image_slice.cpu().numpy().T, cmap='gray', origin='lower')
    axes[1].imshow(np.ma.masked_where(gt_slice.cpu().numpy() == 0, gt_slice.cpu().numpy()).T,
                   cmap='autumn', alpha=0.5, origin='lower')
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Prediction Overlay
    axes[2].imshow(image_slice.cpu().numpy().T, cmap='gray', origin='lower')
    axes[2].imshow(np.ma.masked_where(pred_slice.cpu().numpy() == 0, pred_slice.cpu().numpy()).T,
                   cmap='winter', alpha=0.5, origin='lower')
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()