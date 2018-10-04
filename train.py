import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

USE_CUDA = torch.cuda.device_count() >= 1

def convert_to_torch_tensor(data_X, data_Y):
	X, Y = torch.from_numpy(data_X), torch.from_numpy(data_Y)
	if USE_CUDA:
            X = X.cuda()
            Y = Y.cuda()
	return X, Y

def train(model, data_X, data_Y, n_epochs = 20, learning_rate = 0.01, print_after_every = 2):
	if USE_CUDA:
            model=model.cuda()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	model.train()
	data_X, data_Y = convert_to_torch_tensor(data_X, data_Y)

	for epoch in range(n_epochs):
		for batch_idx in range(data_X.shape[0]):
			X = Variable(data_X[batch_idx])
			Y = torch.max(Variable(data_Y[batch_idx]), 1)[1]
			optimizer.zero_grad()
			prediction = model(X)
			loss = F.cross_entropy(prediction, Y)
			loss.backward()
			optimizer.step()
			if batch_idx % print_after_every == 0:
				print('Train Epoch: {} Batch Index: {} Total Number of Batches: {} \tLoss: {:.6f}'.format(
					epoch, batch_idx , data_X.shape[0], loss.item()))

def test(model, data_X, data_Y):
	if USE_CUDA:
            model=model.cuda()
	model.eval()
	data_X, data_Y = convert_to_torch_tensor(data_X, data_Y)
	loss= 0
	correct = 0
	for batch_idx in range(data_X.shape[0]):
		X = Variable(data_X[batch_idx])
		Y = torch.max(Variable(data_Y[batch_idx]), 1)[1]
		prediction = model(X)
		loss += F.cross_entropy(prediction, Y, reduction='sum').item()
		prediction = prediction.data.max(1, keepdim=True)[1]
		correct += prediction.eq(Y.data.view_as(prediction)).cpu().sum()
	data_len = data_X.shape[0] * data_X.shape[1]
	loss /= data_len
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		loss, correct, data_X.shape[0] * data_X.shape[1],
		100. * correct / data_len))




