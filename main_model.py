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


        self.model1 = m[0]
        self.model2 = m[1]
        self.model3 = m[2]
        #self.model4 = m[3]
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
        x1 = self.model1(x[0])
        x2 = self.model2(x[1])
        x3 = self.model3(x[2])
        #x4 = self.model4(x[3])
        x = torch.cat((x1,x2,x3),dim = -1)
        x = self.fc(x)
        return x

    def name(self):
        return "MainModel"