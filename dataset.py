import torch
import torch.utils.data
from torch.utils.data import Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision import datasets
from augmentation import get_augmentation
## Import Data Loaders ##
from dataloader import *

python_file_name= 'blur'


def get_dataset(dataset, batch, imsize, workers):
    if dataset == 'G':
        train_dataset = GTA5(list_path='./data_list/GTA5', split='train', crop_size=imsize)
        test_dataset = None

    elif dataset == 'C':
        train_dataset = Cityscapes(list_path='./data_list/Cityscapes', split='train', crop_size=imsize)
        test_dataset = Cityscapes(list_path='./data_list/Cityscapes', split='val', crop_size=imsize, train=False)

    elif dataset == 'M':
        train_transform = get_augmentation(python_file_name,imsize)
        print(train_transform)
        test_transform = transforms.Compose([
                                    transforms.Resize(imsize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
        initial_train_dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=train_transform)
        initial_test_dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=train_transform)
        targets = np.array(initial_train_dataset.targets)
        train_indices = []
        test_indices = []
        for class_idx in range(10):
            class_indices = np.where(targets == class_idx)[0]
            selected_indices = np.random.choice(class_indices, 1000, replace=False)
            selected_test_indices = np.random.choice(class_indices, 200, replace=False)
            train_indices.extend(selected_indices)
            test_indices.extend(selected_test_indices)
        train_dataset = Subset(initial_train_dataset, train_indices)
        test_dataset = Subset(initial_test_dataset, test_indices)


                                  
    elif dataset == 'U':
        train_dataset = dset.USPS(root='./data/usps', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(imsize),
                                       transforms.Grayscale(3),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        test_dataset = dset.USPS(root='./data/usps', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize(imsize),
                                      transforms.Grayscale(3),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
    elif dataset == 'MM':
        test_transform = transforms.Compose([
                                    transforms.Resize(imsize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
        initial_train_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=test_transform)
        initial_test_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=test_transform)
        targets = np.array(initial_train_dataset.targets)
        train_indices = []
        test_indices = []
        for class_idx in range(10):
            class_indices = np.where(targets == class_idx)[0]
            selected_indices = np.random.choice(class_indices, 500, replace=False)
            selected_test_indices = np.random.choice(class_indices, 200, replace=False)
            train_indices.extend(selected_indices)
            test_indices.extend(selected_test_indices)
        train_dataset = Subset(initial_train_dataset, train_indices)
        test_dataset = Subset(initial_test_dataset, test_indices)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch,
                                                   shuffle=True, num_workers=int(workers), pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch*4,
                                                   shuffle=False, num_workers=int(workers))
    return train_dataloader, test_dataloader
