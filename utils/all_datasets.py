import os
import torch
from torchvision import transforms
from torchvision import datasets 
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import CIFAR10

def imagenet(path, split='val'):
    return ImageNet_Dataset(path, split)

def cifar10(path, split='val'):
    return CIF10_Dataset(path, split)
    

class CIF10_Dataset(Dataset):
    def __init__(self, path, split):
        self.tensors = CIFAR10(root=path,
                               download=True,
                               train=split=='train',
                               transform=transforms.ToTensor())

    def __getitem__(self, index):
        data, target = self.tensors[index]
        return index, data, target

    def __len__(self):
        return len(self.tensors)

class ImageNet_Dataset(Dataset):
    def __init__(self, path, split):
        subdir = os.path.join(path, split)
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
        self.tensors = datasets.ImageFolder(subdir, transform)

    def __getitem__(self, index):
        data, target = self.tensors[index]
        return index, data, target

    def __len__(self):
        return len(self.tensors)

def random_dataset(num_instances, num_classes, resolution):
    x = torch.randn((num_instances, 3, resolution, resolution))
    y = torch.randint(0, num_classes, (num_instances,)) 
    i = torch.arange(num_instances)
    print('Creating a random dataset with {} instances, {} classes, and {} input resolution'.format(
        num_instances, num_classes, resolution))
    return TensorDataset(i, x, y)


class Fake_Dataset(Dataset):
    def __init__(self, path):
        self.tensors = datasets.ImageFolder(path, transforms.ToTensor())

    def __getitem__(self, index):
        data, target = self.tensors[index]
        return index, data, target

    def __len__(self):
        return len(self.tensors)