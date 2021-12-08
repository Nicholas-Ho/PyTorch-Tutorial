from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule

# Pytorch Lightning operates on pure dataloaders
class MNISTDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.download_dir = os.getcwd()
        self.batch_size = 64
        
        # Transforms - prepare transforms standard to MNIST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    # Called on only 1 GPU
    def prepare_data(self):
        datasets.MNIST(self.download_dir, train=True, download=True)
        datasets.MNIST(self.download_dir, train=False, download=True)
        
    # Called on every GPU
    def setup(self, stage=None):
        data = datasets.MNIST(self.download_dir, train=True, transform=self.transform)
        
        self.train, self.val = random_split(data, [55000, 5000])
        self.test = datasets.MNIST(self.download_dir, train=False, transform=self.transform)
        
    # Dataloaders for training, validation and testing
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)