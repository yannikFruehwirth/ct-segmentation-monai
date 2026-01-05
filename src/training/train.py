# -- IMPORTS --
import os
import matplotlib
from monai.apps import download_and_extract
import torch
from monai.losses import DiceLoss
from monai.networks.utils import one_hot
from monai.inferers import SlidingWindowInferer
from monai.transforms import AsDiscrete, Compose
from monai.data import decollate_batch
from tqdm import tqdm

# -- CODE --
def train_one_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    step_loss = 0
    
    for batch_data in loader:
        inputs = batch_data["image"]
        labels = batch_data["label"]
        
        # Batch Dim Handling
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(0)
        if labels.dim() == 4:
            labels = labels.unsqueeze(0)
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        step_loss += loss.item()
        
    return step_loss / len(loader)

def validate_one_epoch(model, loader, dice_metric, device):
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    
    post_label = AsDiscrete(to_onehot=2)

    inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.25)
    
    model.eval()
    dice_metric.reset()
    
    with torch.no_grad():
        with tqdm(loader, desc="Validation", unit="batch") as pbar:
            for batch_data in pbar:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                
                if inputs.dim() == 4:
                    inputs = inputs.unsqueeze(0)
                if labels.dim() == 4:
                    labels = labels.unsqueeze(0)

                outputs = inferer(inputs, model)
                
                # 2. Decollate Batch
                val_outputs_list = decollate_batch(outputs)
                val_labels_list = decollate_batch(labels)
                
                # 3. Post-Processing 
                
                val_outputs_processed = [post_pred(i) for i in val_outputs_list]
                val_labels_processed = [post_label(i) for i in val_labels_list]
                
                # 4. metrics
                dice_metric(y_pred=val_outputs_processed, y=val_labels_processed)
            
            mean_dice = dice_metric.aggregate().item()
            pbar.set_postfix(dice=f"{mean_dice:.4f}")
            
    return mean_dice