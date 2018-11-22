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
        m = []
        for key, val in model_dict.items():
            m.append(getattr(sys.modules[__name__],key)(val[0],val[1]))
            inputs += val[1]


        self.models = m
        self.fc = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(inputs, 2048),
                                nn.ReLU(),
                                 nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Linear(512,128),
                                 nn.ReLU(),
                                 nn.Linear(128,num_classes))

    def forward(self, x):
        output = []
        for idx, model in enumerate(self.models):
            output.append(model(x[idx]))
       
        x = torch.cat(output,dim = -1)


        x = self.fc(x)
        return x

    def name(self):
        return "MainModel"