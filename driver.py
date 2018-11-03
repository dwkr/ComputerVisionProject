import math
import numpy as np
import glob
from balanceData import balance_data, gen_stats
import argparse
from visualizeData import visualize_raw_data, visualize_batch_data
import random
from random import shuffle
import models
from train import train, test
import logging
from datetime import datetime
from extract_features import getFeatures
import json
import sys
from main_model import MainModel

with open("config2.json",'r') as file:
    config = json.load(file)

parser = argparse.ArgumentParser(description="Autonomous Driving for GTA 5")

parser.add_argument('--num_epochs', default=config['num_epochs'], type=int,
    help='number of training epochs')

parser.add_argument('--batch_size', type=int, default=config['batch_size'],
    help='Batch Size')

parser.add_argument('--gpu', action='store_true', default=config['gpu'],
    help='Use GPU')

parser.add_argument('--gpu_number', default=config['gpu_number'], type=int,
    help='Which GPU to run on')

parser.add_argument('--balance', action='store_true', default=config['balance'],
    help='Balance Data')

parser.add_argument('--shuffle_data', action='store_true', default=config['shuffle_data'],
    help='Shuffle Train Data')

parser.add_argument('--lr', type=float, default=config['lr'],
    help='Learning rate')

parser.add_argument('--train_ratio', type=float, default=config['train_ratio'],
    help='Ratio for Training Examples')

parser.add_argument('--validation_ratio', type=float, default=config['validation_ratio'],
    help='Ratio for Validation Examples')

parser.add_argument('--test_ratio', type=float, default=config['test_ratio'],
    help='Ratio for Test Examples')

parser.add_argument('--train_data', default=config['train_data'], type=str,
    help='path to train data')

parser.add_argument('--save_model', default=config['save_model'], type=str,
    help='path to directory to save model weights')

parser.add_argument('--log_dir', default=config['log_dir'], type=str,
    help='path to directory to save logs')

parser.add_argument('--model', default=config['model'], type=str,
    help='Which model to use')

parser.add_argument('--print_after', default=config['print_after'], type=int,
    help='Print Loss after every n iterations')

parser.add_argument('--validate_after', default=config['validate_after'], type=int,
    help='Validate after every n iterations')

parser.add_argument('--save_after', default=config['save_after'], type=int,
    help='Save after every n iterations')

parser.add_argument('--seed', default=config['seed'], type=int,
    help='Random Seed to Set')

parser.add_argument('--print', action='store_true', default=config['print'],
    help='Print Log Output to stdout')

args = parser.parse_args()

random.seed(args.seed)


if args.print:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root = logging.getLogger()
    root.addHandler(ch)

logging.basicConfig(level=logging.INFO,
    filename= args.log_dir + datetime.now().strftime('GTA_%d_%m_%Y_%H_%M_%S.log'),
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

logging.info(args)

if(args.train_ratio + args.test_ratio + args.validation_ratio != 1.0):
    raise ValueError('Sum of Train, Test and Validation Ratios must be 1.0')

def load_data(input_path, balance, shuffe_data = True):
    logging.info("Reading files from {}".format(input_path))
    files = glob.glob(input_path)
    logging.info("Number of files available : {}".format(len(files)))
    all_data = []
    for file in files:
        logging.info("loading file {} ".format(file))
        data = np.load(file)
        features = getFeatures(data)
        #print("Features type",type(features))
        all_data.extend(features)

    

    if balance:
        logging.info("Balancing Data")
        shuffle(all_data)
        stats_data, all_data = balance_data(all_data)
        logging.info("Data Stats for balanced data: {}".format(stats_data))
    else:
        logging.info("Not Balancing Data")
        stats_data = gen_stats(all_data)
        logging.info("Data Stats for unblanced data: {}".format(stats_data))

    if(shuffe_data):
        shuffle(all_data)

    logging.info("Data Read successfully")
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
    X1 = np.array([i[0] for i in data], dtype=np.float32).reshape(n_batches,batch_size,3,299,299)
    X2 = np.array([i[2] for i in data], dtype=np.float32).reshape(n_batches,batch_size,1,64,64)
    X3 = np.array([i[3] for i in data], dtype=np.float32).reshape(n_batches,batch_size,1,64,64)
    X4 = np.array([i[4] for i in data], dtype=np.float32).reshape(n_batches,batch_size,100,5)
    Y = np.array([i[1] for i in data]).reshape(n_batches, batch_size, 5)
    return X1, X2, X3, X4, Y

def train_AI(input_path, model_save_path):
    data = load_data(input_path, balance = args.balance, shuffe_data = args.shuffle_data)
    batched_data_X1, batched_data_X2, batched_data_X3, batched_data_X4, batched_data_Y = create_batches(data, args.batch_size)
    train_data_X1, validation_data_X1, test_data_X1 = create_sets(batched_data_X1, args.train_ratio, args.validation_ratio, args.test_ratio)
    train_data_X2, validation_data_X2, test_data_X2 = create_sets(batched_data_X2, args.train_ratio, args.validation_ratio, args.test_ratio)
    train_data_X3, validation_data_X3, test_data_X3 = create_sets(batched_data_X3, args.train_ratio, args.validation_ratio, args.test_ratio)
    train_data_X4, validation_data_X4, test_data_X4 = create_sets(batched_data_X4, args.train_ratio, args.validation_ratio, args.test_ratio)
    train_data_X = [train_data_X1, train_data_X2, train_data_X3, train_data_X4]
    validation_data_X = [validation_data_X1, validation_data_X2, validation_data_X3, validation_data_X4]
    test_data_X = [test_data_X1, test_data_X2, test_data_X3, test_data_X4]
    train_data_Y, validation_data_Y, test_data_Y = create_sets(batched_data_Y, args.train_ratio, args.validation_ratio, args.test_ratio)

    logging.info("Number of Training Examples : {}".format(train_data_X1.shape[0] * train_data_X1.shape[1]))
    logging.info("Number of Validation Examples : {}".format(validation_data_X1.shape[0] * validation_data_X1.shape[1]))
    logging.info("Number of Test Examples : {}".format(test_data_X1.shape[0] * test_data_X1.shape[1]))

    logging.info("Creating Model Graph")
    model = getattr(sys.modules[__name__],args.model)(config['model_dict'],5)
    logging.info("Model Created successfully")

    logging.info("Starting Training")
    train(logging, model, config['loss_weights'], train_data_X, train_data_Y, validation_data_X, validation_data_Y,  args.num_epochs, args.lr, args.gpu, args.gpu_number, args.save_model, args.print_after, args.validate_after, args.save_after)
    logging.info("Training Completed")
    
    if(validation_data_X.shape[0] * validation_data_X.shape[1] > 0):
        logging.info("Testing on Validation Set")
        test(logging, model, config['loss_weights'], validation_data_X, validation_data_Y, args.gpu)
    
    if( test_data_X.shape[0] * test_data_X.shape[1] > 0):
        logging.info("Testing on Test Set")
        test(logging, model, config['loss_weights'], test_data_X, test_data_Y, args.gpu, args.gpu_number)


if __name__ == "__main__":

    train_AI(args.train_data, args.save_model)

