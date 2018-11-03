import torch.nn as nn
from xception import xception

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