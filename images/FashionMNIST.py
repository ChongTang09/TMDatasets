import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class FashionMNISTDataset(Dataset):
    def __init__(self, root_dir='./root_dir', train=True, transform=None, download=True):

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        self.fashion_mnist_data = datasets.FashionMNIST(root=root_dir, train=train, download=download)
        
        if transform:
            self.transform = transform
    
    def __len__(self):
        return len(self.fashion_mnist_data)
    
    def __getitem__(self, idx):
        image, label = self.fashion_mnist_data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label