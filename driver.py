import math
import numpy as np
import glob
from balanceData import balance_data, gen_stats
import argparse
import random
from random import shuffle
import models
from train import test_with_loader, trainer
import logging
from datetime import datetime
from extract_features import getFeatures
import json
import sys
from main_model import MainModel
from GTA_data_set import GTADataset
from GTA_clip_data_set import GTAClipDataSet
import torch
from find_stats import findStats
import pprint
import re

config_file_path = "configs/config.json"

with open(config_file_path,'r') as file:
    config = json.load(file)

parser = argparse.ArgumentParser(description="Autonomous Driving for GTA 5")

parser.add_argument('--num_epochs', default=config['num_epochs'], type=int,
    help='number of training epochs')

parser.add_argument('--batch_size', type=int, default=config['batch_size'],
    help='Batch Size')

parser.add_argument('--clip_len', type=int, default=config['clip_len'],
    help='Clip Length')

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

parser.add_argument('--log_name', default=config['log_name'], type=str,
    help='name of the log file starting string')

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

parser.add_argument('--cal_stats', action='store_true', default=config['cal_stats'],
    help='Caclulate stats for normalisation')

args = parser.parse_args()

random.seed(args.seed)


logging.basicConfig(level=logging.INFO,
    filename= args.log_dir + args.log_name + datetime.now().strftime('%d_%m_%Y_%H_%M_%S.log'),
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

if args.print:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(ch)


logging.info("Reading config from : {}".format(config_file_path))

logging.info(args)

logging.info("Config : {}".format(json.dumps(config, indent=4)))

if(args.train_ratio + args.test_ratio + args.validation_ratio != 1.0):
    raise ValueError('Sum of Train, Test and Validation Ratios must be 1.0')

def load_data(input_path, balance, shuffe_data = True):
    logging.info("Reading files from {}".format(input_path))
    files = glob.glob(input_path)
    files.sort()
    logging.info("Number of files available : {}".format(len(files)))
    all_data = []
    for idx, file in enumerate(files):
        logging.info("loading file {}, {}/{} loaded".format(file, idx, len(files)))
        data = np.load(file)
        features = getFeatures(data)
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
    n_batches = len(data)
    data = data[:n_batches*batch_size]
    X1 = np.array([i[0] for i in data], dtype=np.float32).reshape(n_batches,3,299,299)
    X2 = np.array([i[2] for i in data], dtype=np.float32).reshape(n_batches,1,64,64)
    X3 = np.array([i[3] for i in data], dtype=np.float32).reshape(n_batches,1,64,64)
    X4 = np.array([i[4] for i in data], dtype=np.float32).reshape(n_batches,100,5)
    Y = np.array([i[1] for i in data]).reshape(n_batches,  5)
    return X1, X2, X3, X4, Y
        
        
    
def create_loader_sets(data, train_ratio, validation_ratio, test_ratio, stats, normalize):
    train_len = int(len(data) * train_ratio)
    validation_len = int(len(data) * validation_ratio)

    train_dataset = GTADataset(data,0, train_ratio, stats, normalize)
    validation_dataset = GTADataset(data, len(train_dataset), validation_ratio, stats, normalize)
    test_dataset = GTADataset(data, len(train_dataset) + len(validation_dataset), test_ratio, stats, normalize)

    return train_dataset, validation_dataset, test_dataset   
    
def train_AI_with_loaders(input_path, model_save_path):
    #data = load_data(input_path, balance = args.balance, shuffe_data = args.shuffle_data)
    
    #stat_dataset, _ ,_ = create_loader_sets(data, args.train_ratio, args.validation_ratio, args.test_ratio, None, False)
    #stat_loader = torch.utils.data.DataLoader(stat_dataset,batch_size = args.batch_size, shuffle = True, num_workers = 0)
    
    #stats = findStats(logging, stat_loader, args.cal_stats)
    #stat_dataset = None
    #stat_loader = None
    
    #train_dataset, val_dataset, test_dataset = create_loader_sets(data, args.train_ratio, args.validation_ratio, args.test_ratio, stats, True)

    #train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = args.batch_size, shuffle = True, num_workers = 0)
    #val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = args.batch_size, shuffle = False, num_workers = 0)
    #test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = args.batch_size, shuffle = False, num_workers = 0)

    train_files = list(filter(lambda i : int(re.search('train_data_(.+?).npy', i).group(1)) <= 480,
                              glob.glob(args.train_data)))
    val_files = list(filter(lambda i: int(re.search('train_data_(.+?).npy', i).group(1)) < 500 and int(re.search('train_data_(.+?).npy', i).group(1)) > 480,
                              glob.glob(args.train_data)))

    test_files = list(filter(lambda i: int(re.search('train_data_(.+?).npy', i).group(1)) < 520 and int(re.search('train_data_(.+?).npy', i).group(1)) > 500,
                              glob.glob(args.train_data)))



    train_dataset = GTAClipDataSet(args.train_data, args.clip_len, findStats(None, None, False), logging, files=train_files)
    val_dataset = GTAClipDataSet(args.train_data, args.clip_len, findStats(None, None, False), logging, files=val_files)
    test_dataset = GTAClipDataSet(args.train_data, args.clip_len, findStats(None, None, False), logging, files=test_files)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logging.info("Number of Training Examples : {}".format(len(train_dataset)))
    logging.info("Number of Validation Examples : {}".format(len(val_dataset)))
    logging.info("Number of Test Examples : {}".format(len(test_dataset)))
    
    logging.info("Creating Model Graph")
    model = getattr(sys.modules[__name__],args.model)(config['model_dict'], 5)
    logging.info("Model Created successfully")

    logging.info("Starting Training")
    trainer(logging, model, config['loss_weights'], train_loader, val_loader,  args.num_epochs, args.lr, args.gpu, args.gpu_number, args.save_model, args.print_after, args.validate_after, args.save_after)
    logging.info("Training Completed")
    
    if(len(val_loader) > 0):
        logging.info("Testing on Validation Set")
        test_with_loader(logging, model, config['loss_weights'], val_loader, args.gpu, args.gpu_number)
    
    if(len(test_loader) > 0):
        logging.info("Testing on Test Set")
        test_with_loader(logging, model, config['loss_weights'], test_loader, args.gpu, args.gpu_number)


if __name__ == "__main__":

    train_AI_with_loaders(args.train_data, args.save_model)
   # train_AI(args.train_data, args.save_model)

