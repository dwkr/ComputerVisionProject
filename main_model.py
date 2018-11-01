import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from xception import xception
from image_model import *
from map_model import *
from speed_model import *
from history_model import *
import sys

class MainModel(nn.Module):
    """docstring for MainModel"""
    def __init__(self, model_dict, num_classes):
        super(MainModel, self).__init__()

        inputs = 0
        self.models = []
        for key, val in model_dict.items():
            self.models.append(getattr(sys.modules[__name__],key)(val[0],val[1]))
            inputs += val[1]
        
        self.fc = nn.Sequential(nn.Linear(inputs, 2048),
                                 nn.Linear(2048, 512),
                                 nn.Linear(512,128),
                                 nn.Linear(128,num_classes))

    def forward(self, x):
        x1 = self.models[0](x[0])
        x2 = self.models[1](x[1])
        x3 = self.models[2](x[2])
        x4 = self.models[3](x[3])

        x = torch.cat((x1,x2,x3,x4),dim = -1)

        x = self.fc(x)
        return x

    def name(self):
        return "MainModel"