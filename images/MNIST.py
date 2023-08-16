import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class MNIST(Dataset):
    def __init__(self, root_dir='./root_dir', train=True, transform=None, download=True):
        self.mnist_data = datasets.MNIST(root=root_dir, train=train, download=download)

        if transform:
            self.transform = transform
    
    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label