import torch.nn as nn

class HistoryModel(nn.Module):
    """docstring for HistoryModel"""
    def __init__(self,num_inputs, num_outputs):
       super(HistoryModel, self).__init__()
       self.model = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
       x = x.view(-1,x.shape[1]*x.shape[2])
       x = self.model(x)
       return x

    def name(self):
       return "HistoryModel"