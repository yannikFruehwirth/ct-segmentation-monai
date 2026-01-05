# -- IMPORTS -- 
import os
import matplotlib
from monai.apps import download_and_extract
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
    Lambdad,
    MapLabelValued,
    SpatialPadd
)

# -- CODE --
def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        MapLabelValued(keys=["label"], orig_labels=[2], target_labels=[1]),

        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(96, 96, 96), 
            random_center=True,
            random_size=False
        ),
        
        # Augmentations
        RandRotated(keys=["image", "label"], range_x=0.3, prob=0.5, mode=("bilinear", "nearest")),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.1),
        
        EnsureTyped(keys=["image", "label"])
    ])

    
def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        MapLabelValued(keys=["label"], orig_labels=[2], target_labels=[1]),
        
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=["image", "label"])
    ])