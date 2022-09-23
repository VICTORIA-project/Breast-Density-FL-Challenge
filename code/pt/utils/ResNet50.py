from torchvision import models
import torch.nn as nn
import torch


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=4, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x