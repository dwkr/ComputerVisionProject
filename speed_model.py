import torch.nn as nn
from torchvision import models

class SpeedResNet50(nn.Module):
    """docstring for SpeedResNet50"""
    def __init__(self,num_inputs, num_outputs):
       super(SpeedResNet50, self).__init__()
       self.model = models.resnet50()
       self.model.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
       self.model.avgpool = nn.AvgPool2d(2, stride=1)
       self.model.fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
       self.fc2 = nn.Linear(in_features=1024, out_features=num_outputs, bias=True)

    def forward(self, x):
       x = self.model(x)
       x = self.fc2(x)
       return x

    def name(self):
       return "SpeedResNet50"