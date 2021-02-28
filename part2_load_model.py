from torchvision import models
import torch.nn as nn
import torch
from CONFIG import CONFIG

class loadmodel:
    def __init__(self, classes: int):
        self.model = models.vgg16(pretrained=True)

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, classes)
        # print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")

        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")
        if CONFIG["GPU"]:
            self.device = torch.device("cuda")
            self.model.device = self.device

    def print_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")

        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")


