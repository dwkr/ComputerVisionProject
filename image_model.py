import torch.nn as nn
from xception import xception
from torchvision import models

class Image_Xception(nn.Module):
    """docstring for Image_Xception"""
    def __init__(self,num_inputs, num_outputs):
       super(Image_Xception, self).__init__()
       self.model = xception(num_classes=num_outputs)
       self.model.fc = nn.Linear(2048, num_outputs)

    def forward(self, x):
       x = self.model(x)
       return x

    def name(self):
       return "Image_Xception"

class ImageResNet50(nn.Module):
    """docstring for ImageResNet50"""
    def __init__(self,num_inputs, num_outputs):
       super(ImageResNet50, self).__init__()
       self.model = models.resnet50()
       self.model.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
       self.model.fc = nn.Linear(in_features=2048, out_features=num_outputs, bias=True)

    def forward(self, x):
       x = self.model(x)
       return x

    def name(self):
       return "ImageResNet50"

class ImageResNet101(nn.Module):
    """docstring for ImageResNet101"""
    def __init__(self,num_inputs, num_outputs):
       super(ImageResNet101, self).__init__()
       self.model = models.resnet101()
       self.model.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
       self.model.fc = nn.Linear(in_features=2048, out_features=num_outputs, bias=True)

    def forward(self, x):
       x = self.model(x)
       return x

    def name(self):
       return "ImageResNet101"

class ImageDenseNet121(nn.Module):
    """docstring for ImageDenseNet121"""
    def __init__(self,num_inputs, num_outputs):
       super(ImageDenseNet121, self).__init__()
       self.model = models.densenet121()
       self.model.features.conv0 = nn.Conv2d(num_inputs, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
       self.model.classifier = nn.Linear(in_features=1024, out_features=num_outputs, bias=True)

    def forward(self, x):
       x = self.model(x)
       return x

    def name(self):
       return "ImageDenseNet121"
