import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

class load_data():
    def __init__(self, args) -> None:
            traindir = "./data/15SceneData/train"
            valdir = "./data/15SceneData/test"
            train_data_transform = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
            train_dataset = datasets.ImageFolder(traindir, transform=train_data_transform)
            val_data_transform = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
            val_dataset = datasets.ImageFolder(valdir, transform=val_data_transform)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

