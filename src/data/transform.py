# -- IMPORTS -- 
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
    Resized,
    DivisiblePadd,
    RandSpatialCropd,       
    RandRotated,            
    RandFlipd,              
    RandGaussianNoised,
)

# -- CODE --

def get_train_transforms():
    return Compose([
        # 1. Loading
        LoadImaged(keys=["image", "label"]),
        
        # 2. channel handling
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # 3. same orientation for all data
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # 4. Spacing due to anisotropy (see analysis)
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.5, 1.5, 1.5), 
            mode=("bilinear", "nearest") 
        ),
        
        # 5. windowing & scaling
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=-100, a_max=250,
            b_min=0.0, b_max=1.0, 
            clip=True
        ),
        
        # 6. cropping for efficiency
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # 7. augmentation
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(96, 96, 96),
            random_center=True,
            random_size=False
        ),
        RandRotated(
            keys=["image", "label"],
            range_x=0.26, range_y=0.26, range_z=0.26, # ca. 15 Grad
            prob=0.5,
            mode=("bilinear", "nearest")
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
        
        # 8. divisible padding to prevent dim conflict in unet skip connections
        DivisiblePadd(keys=["image", "label"], k=16),
        
        # 9. Resizing patches to ensure constant patch size
        Resized(
            keys=["image", "label"], 
            spatial_size=(128, 128, 128), 
            mode=("trilinear", "nearest")
        ),
        
        # 10. to tensor
        EnsureTyped(keys=["image", "label"])
    ])
    
def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
        EnsureTyped(keys=["image", "label"])
    ])