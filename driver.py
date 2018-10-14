import math
import numpy as np
import glob
from balanceData import balance_data
import argparse
from visualizeData import visualize_raw_data, visualize_batch_data
from random import shuffle
import models
from train import train, test
import torch

parser = argparse.ArgumentParser(description="Autonomous Driving for GTA 5")

parser.add_argument('--num_epochs', default=100, type=int,
	help='number of training epochs')

parser.add_argument('--batch_size', type=int, default=32,
	help='Batch Size')

parser.add_argument('--gpu', action='store_true', default=False,
	help='Use GPU')

parser.add_argument('--gpu_number', default=0, type=int,
	help='Which GPU to run on')

parser.add_argument('--balance', action='store_true', default=False,
	help='Balance Data')

parser.add_argument('--lr', type=float, default=3e-4,
	help='Learning rate')

parser.add_argument('--train_ratio', type=float, default=0.7,
	help='Ratio for Training Examples')

parser.add_argument('--validation_ratio', type=float, default=0.1,
	help='Ratio for Validation Examples')

parser.add_argument('--test_ratio', type=float, default=0.2,
	help='Ratio for Test Examples')

parser.add_argument('--train_data', default='../input/*.npy', type=str,
	help='path to train data')

parser.add_argument('--save_model', default='../models/', type=str,
	help='path to directory to save model weights')

parser.add_argument('--model', default='SimpleConvNet', type=str,
	help='Which model to use')

parser.add_argument('--print_after', default=1, type=int,
	help='Print Loss after every n iterations')

parser.add_argument('--validate_after', default=1, type=int,
	help='Validate after every n iterations')

parser.add_argument('--save_after', default=1, type=int,
	help='Save after every n iterations')

args = parser.parse_args()
print(args)



def selectModel():
	if args.model == "SimpleConvNet":
		return models.SimpleConvNet(9)

def load_data(input_path, balance, shuffe_data = True):
	print("Reading files from ", input_path)
	files = glob.glob(input_path)
	print("Number of files available : ", len(files))
	all_data = []
	for file in files:
		print("loading file", file)
		data = np.load(file)
		all_data.extend(data)

	if balance:
		print("Balancing Data")
		shuffle(all_data)
		_, all_data = balance_data(all_data)
		print("Data Stats : ", _)
	
	if(shuffe_data):
		shuffle(all_data)

	print("Data Read successfully")
	return all_data

def create_sets(data, train_ratio, validation_ratio, test_ratio):
	train_len = int(data.shape[0] * train_ratio)
	validation_len = int(data.shape[0] * validation_ratio)

	train_data = data[:train_len]
	validation_data = data[train_len:train_len+validation_len]
	test_data = data[train_len+validation_len : ]

	return train_data, validation_data, test_data

def create_batches(data, batch_size):
	n_batches = int(math.floor(len(data)/batch_size))
	data = data[:n_batches*batch_size]
	X = np.array([i[0] for i in data], dtype=np.float32).reshape(n_batches,batch_size,299,299,3)
	Y = np.array([i[1] for i in data]).reshape(n_batches, batch_size, 9)
	return X,Y

def train_AI(input_path, model_save_path):
	data = load_data(input_path, balance = args.balance, shuffe_data = True)
	batched_data_X, batched_data_Y = create_batches(data, args.batch_size)
	train_data_X, validation_data_X, test_data_X = create_sets(batched_data_X, args.train_ratio, args.validation_ratio, args.test_ratio)
	train_data_Y, validation_data_Y, test_data_Y = create_sets(batched_data_Y, args.train_ratio, args.validation_ratio, args.test_ratio)

	print("Number of Training Examples : ", train_data_X.shape[0] * train_data_X.shape[1])
	print("Number of Validation Examples : ", validation_data_X.shape[0] * validation_data_X.shape[1])
	print("Number of Test Examples : ", test_data_X.shape[0] * test_data_X.shape[1])

	print("Creating Model Graph")
	model = selectModel()
	print("Model Created successfully")

	print("Starting Training")
	train(model, train_data_X, train_data_Y, validation_data_X, validation_data_Y,  args.num_epochs, args.lr, args.gpu, args.gpu_number, args.save_model, args.print_after, args.validate_after, args.save_after)
	print("Training Completed")
	
	if(validation_data_X.shape[0] * validation_data_X.shape[1] > 0):
	    print("Testing on Validation Set")
	    test(model, validation_data_X, validation_data_Y, args.gpu)
	
	if( test_data_X.shape[0] * test_data_X.shape[1] > 0):
	    print("Testing on Test Set")
	    test(model, test_data_X, test_data_Y, args.gpu, args.gpu_number)


	print("Saving model to ", model_save_path)
	torch.save(model.state_dict(), model_save_path)
	print("Model saved Successfully")


if __name__ == "__main__":
	train_AI(args.train_data, args.save_model)

