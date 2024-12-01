# augmentation.py
import numpy as np
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
    
    else:
        raise ValueError("Invalid augmentation type")
