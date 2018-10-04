import math
import numpy as np
import glob
from balanceData import balance_data
#from visualizeData import visualize_raw_data, visualize_batch_data
from random import shuffle
import models
from train import train, test
import torch

def load_data(input_path, balance = True, shuffe_data = True):
	print("Reading files from ", input_path)
	files = glob.glob(input_path)
	print("Number of files available : ", len(files))
	all_data = []
	for file in files:
		data = np.load(file)
		if balance:
			_, balanced_data = balance_data(data)
		else:
			balanced_data = data
		if(shuffe_data):
			shuffle(balanced_data)
		all_data.extend(balanced_data)
	if(shuffe_data):
		shuffle(all_data)
	print("Data Read successfully")
	return all_data

def create_sets(data, train_ratio = 0.7, validation_ratio = 0.1, test_ratio = 0.2):
	train_len = int(data.shape[0] * train_ratio)
	validation_len = int(data.shape[0] * validation_ratio)

	train_data = data[:train_len]
	validation_data = data[train_len:train_len+validation_len]
	test_data = data[train_len+validation_len : ]

	return train_data, validation_data, test_data

def create_batches(data, batch_size = 32):
	n_batches = int(math.floor(len(data)/batch_size))
	data = data[:n_batches*batch_size]
	X = np.array([i[0] for i in data], dtype=np.float32).reshape(n_batches,batch_size,299,299,3)
	Y = np.array([i[1] for i in data]).reshape(n_batches, batch_size, 9)
	return X,Y

def train_AI(input_path, model_save_path):
	data = load_data(input_path, balance = True, shuffe_data = True)
	batched_data_X, batched_data_Y = create_batches(data)
	train_data_X, validation_data_X, test_data_X = create_sets(batched_data_X)
	train_data_Y, validation_data_Y, test_data_Y = create_sets(batched_data_Y)

	print("Number of Training Examples : ", train_data_X.shape[0] * train_data_X.shape[1])
	print("Number of Validation Examples : ", validation_data_X.shape[0] * validation_data_X.shape[1])
	print("Number of Test Examples : ", test_data_X.shape[0] * test_data_X.shape[1])

	print("Creating Model Graph")
	model = models.SimpleConvNet(9)
	print("Model Created successfully")

	print("Starting Training")
	train(model, train_data_X, train_data_Y, 20, 0.01, 1)
	print("Training Completed")

	print("Testing on Validation Set")
	test(model, validation_data_X, validation_data_Y)


	print("Saving model to ", model_save_path)
	torch.save(model.state_dict(), model_save_path)
	print("Model saved Successfully")


if __name__ == "__main__":
	train_AI('/data/ra2630/CV/Project/input/*.npy', 
		'/data/ra2630/CV/Project/models/SimpleConvNet')

