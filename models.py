import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

class SimpleConvNet(nn.Module):
	"""docstring for SimpleConvNet"""
	def __init__(self, num_outputs):
		super(SimpleConvNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 5)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, 5)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, 5)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 128, 5)
		self.bn4 = nn.BatchNorm2d(128)
		self.fc1 = nn.Linear(128 * 14 * 14, 64)
		self.fc2 = nn.Linear(64, num_outputs)
		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()

	def forward(self, input):
		input = input.view(-1, 3, 299, 299) # reshape input to batch x num_inputs
		
		output = F.max_pool2d(self.tanh(self.bn1(self.conv1(input))), (2, 2))
		output = F.max_pool2d(self.tanh(self.bn2(self.conv2(output))), (2, 2))
		output = F.max_pool2d(self.relu(self.bn3(self.conv3(output))), (2, 2))
		output = F.max_pool2d(self.relu(self.bn4(self.conv4(output))), (2, 2))
		output = output.view(-1, self.num_flat_features(output))
		output = self.tanh(self.fc1(output))
		output = self.fc2(output)
		return output

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
		    num_features *= s
		return num_features


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
        	nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        
        x = x.view(-1, 3, 299, 299)
        x = self.features(x)
        x = x.view(-1, 256 * 8 * 8)
        x = self.classifier(x)
        return x

    def name(self):
        return "AlexNet"