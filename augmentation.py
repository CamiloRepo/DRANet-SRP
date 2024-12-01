# augmentation.py
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#transforms.Normalize(mean.tolist(), std.tolist())
#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#def get_augmentation(augmentation_type,mean,std):
def get_augmentation(augmentation_type, imsize):
    if augmentation_type == 'withoutda':
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'blur':
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'brightness':
        brightness_factor = 0.3
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.ColorJitter(brightness=brightness_factor),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'colorjitter':
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'contrast':
        contrast_factor = 0.3
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.ColorJitter(contrast=contrast_factor),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'random_affine':
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'rotation':
        angle=10
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.RandomRotation(degrees=angle),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'saturation':
        saturation_factor = 0.5
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.ColorJitter(saturation=saturation_factor),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'shear':
        shear = 10
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.RandomAffine(degrees=0, shear=shear),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'random_transform':
        return transforms.Compose([
            transforms.Resize(imsize),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif augmentation_type == 'elastic_transform':
        return transforms.Compose([
         transforms.Resize(imsize),   
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    elif augmentation_type == 'random_invert':
        return transforms.Compose([
            transforms.Resize(imsize),
        transforms.RandomInvert(p=0.2),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    elif augmentation_type == 'random_posterize':
        return transforms.Compose([
            transforms.Resize(imsize),
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    elif augmentation_type == 'random_solarize':
        return transforms.Compose([
            transforms.Resize(imsize),
        transforms.RandomSolarize(threshold=150, p=0.3),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    elif augmentation_type == 'random_sharpeness':
        return transforms.Compose([
            transforms.Resize(imsize),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    elif augmentation_type == 'random_autocontrast':
        return transforms.Compose([
            transforms.Resize(imsize),
        transforms.RandomAutocontrast(p=0.4),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    elif augmentation_type == 'random_equalize':
        return transforms.Compose([
            transforms.Resize(imsize),
        transforms.RandomEqualize(p=0.4),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    elif augmentation_type == 'grayscale':
        return transforms.Compose([
            transforms.Resize(imsize),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    else:
        raise ValueError("Invalid augmentation type")


"""
{
  "augmentation_techniques": [
    "withoutda",
    "blur",
    "brightness",
    "colorjitter",
    "contrast",
    "elastic_transform",
    "random_affine",
    "random_autocontrast",
    "random_equalize",
    "random_invert",
    "random_posterize",
    "random_sharpeness",
    "random_solarize",
    "random_transform",
    "rotation",
    "saturation",
    "shear",
    "grayscale"
  ]
}
"""