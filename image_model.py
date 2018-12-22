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


class ImageResNet18(nn.Module):
    """docstring for ImageResNet18"""
    def __init__(self,num_inputs, num_outputs):
       super(ImageResNet18, self).__init__()
       model = models.resnet18()
       model.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
       modules = list(models.resnet18().children())[:-2]
       self.model = nn.Sequential(*modules)
       self.avgPoool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)

    def forward(self, x):
       x = self.model(x)
       x = self.avgPoool(x)
       x = x.view(-1, 512)
       return x

    def name(self):
       return "ImageResNet18"


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
       self.avgPoool = nn.AvgPool2d(76, stride=1)
       self.model.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
       self.model.fc = nn.Linear(in_features=2048, out_features=num_outputs, bias=True)

    def forward(self, x):
       x = self.avgPoool(x)
       x = self.model(x)
       return x

    def name(self):
       return "ImageResNet101"

class ImageDenseNet121(nn.Module):
    """docstring for ImageDenseNet121"""
    def __init__(self,num_inputs, num_outputs):
       super(ImageDenseNet121, self).__init__()
       self.model = models.densenet121()
       self.avgPoool = nn.AvgPool2d(76, stride=1)
       self.model.features.conv0 = nn.Conv2d(num_inputs, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
       self.model.classifier = nn.Linear(in_features=1024, out_features=num_outputs, bias=True)

    def forward(self, x):
       x = self.avgPoool(x)
       x = self.model(x)
       return x

    def name(self):
       return "ImageDenseNet121"


class ImageSqueezeNet(nn.Module):
    """docstring for ImageSqueezeNet"""
    def __init__(self,num_inputs, num_outputs):
       super(ImageSqueezeNet, self).__init__()
       self.model = models.squeezenet1_1(num_classes = num_outputs)

    def forward(self, x):
       x = self.model(x)
       x = x.view(x.size(0), 1024)
       return x

    def name(self):
       return "ImageSqueezeNet"