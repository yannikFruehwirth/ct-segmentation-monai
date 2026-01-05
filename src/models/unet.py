# -- IMPORTS --
import os
import matplotlib
from monai.apps import download_and_extract
from monai.networks.nets import UNet

# -- CODE --
def get_unet_model(device):
    model = UNet(
        spatial_dims=3,          # 3d
        in_channels=1,           # 
        out_channels=2,          # bg & target/ pancreas
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),    
        num_res_units=2,         
        norm="batch",            
    ).to(device)
    
    return model