import torch.nn as nn
from torchvision import models

class MapResNet50(nn.Module):
    """docstring for MapResNet50"""
    def __init__(self,num_inputs, num_outputs):
       super(MapResNet50, self).__init__()
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
       return "MapResNet50"
       
class MapEdLeNet(nn.Module):
    """docstring for MapEdLeNet"""
    def __init__(self,num_inputs, num_outputs):
        super(MapEdLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=7, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.AvgPool2d(kernel_size=7)
        )
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64)
        return x

    def name(self):
        return "MapEdLeNet"