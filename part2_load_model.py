from torchvision import models
import torch.nn as nn

class loadmodel:
    def __init__(self, classes: int):
        self.model = models.vgg16(pretrained=True)
        print(self.model)

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, classes)
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")

        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

    def freeze(self, start, stop):
        ct = 0
        for child in self.model.children():
            ct += 1
            if start <=ct < stop:
                for param in child.parameters():
                    param.requires_grad = False


