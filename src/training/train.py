# -- IMPORTS --
import torch
from monai.losses import DiceLoss
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric
from tqdm import tqdm

# -- CODE --
def train_one_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    step_loss = 0
    
    for batch_data in loader:
        inputs = batch_data["image"]
        labels = batch_data["label"]
        
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(0)
        if labels.dim() == 4:
            labels = labels.unsqueeze(0)
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # one hot for dice loss!
        labels_one_hot = one_hot(labels, num_classes=2)
        
        # loss
        loss = loss_function(outputs, labels_one_hot)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        step_loss += loss.item()
        
    return step_loss / len(loader)

def validate_one_epoch(model, loader, dice_metric, device):
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        with tqdm(loader, desc="Validation", unit="batch") as pbar:
            for batch_data in pbar:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                
                # Add batch dimension if missing
                if inputs.dim() == 4:
                    inputs = inputs.unsqueeze(0)
                if labels.dim() == 4:
                    labels = labels.unsqueeze(0)

                outputs = model(inputs)
                
                outputs_softmax = torch.softmax(outputs, dim=1) # necessary for dice loss 
                labels_one_hot = one_hot(labels, num_classes=2)
                
                dice_metric(y_pred=outputs_softmax, y=labels_one_hot)
            
            mean_dice = dice_metric.aggregate().item()
            pbar.set_postfix(dice=f"{mean_dice:.4f}")
            
    return mean_dice
