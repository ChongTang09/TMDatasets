import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir='./root_dir', train=True, transform=None, download=True):
        
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        self.cifar10_data = datasets.CIFAR10(root=root_dir, train=train, download=download)

        # Delete the downloaded .tar.gz file after extracting
        archive = os.path.join(root_dir, 'cifar-10-python.tar.gz')
        if os.path.exists(archive):
            os.remove(archive)
        
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.cifar10_data)
    
    def __getitem__(self, idx):
        image, label = self.cifar10_data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label