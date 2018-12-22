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
from GTA_data_set_sequential import GTADataSetSequential
import torch
from find_stats import findStats
import pprint

config_file_path = "configs/config_112_resnet18_image_only.json"

with open(config_file_path,'r') as file:
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


parser.add_argument('--lr', type=float, default=config['lr'],
    help='Learning rate')

parser.add_argument('--train_data', default=config['train_data'], type=str,
    help='path to train data')

parser.add_argument('--val_data', default=config['val_data'], type=str,
    help='path to test data')

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

logging.info("Config : {}".format(json.dumps(config, indent=4)))

logging.info(args)
        

def train_AI_with_loaders(input_path, model_save_path):
    
    stats = findStats(logging, None, False)

    train_dataset = GTADataSetSequential(args.train_data, stats, logging)
    val_dataset = GTADataSetSequential(args.val_data, stats, logging)
    

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = args.batch_size, shuffle = False, num_workers = 0) #shuffle = true passes shuffled indices passed to get_item
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = args.batch_size, shuffle = False, num_workers = 0)

    logging.info("Number of Training Examples : {}".format(len(train_dataset)))
    logging.info("Number of Validation Examples : {}".format(len(val_dataset)))
    
    logging.info("Creating Model Graph")
    model = getattr(sys.modules[__name__],args.model)(config['model_dict'],5)
    logging.info("Model Created successfully")

    logging.info("Starting Training")
    trainer(logging, model, config['loss_weights'], train_loader, val_loader,  args.num_epochs, args.lr, args.gpu, args.gpu_number, args.save_model, args.print_after, args.validate_after, args.save_after)
    logging.info("Training Completed")
    

if __name__ == "__main__":
    train_AI_with_loaders(args.train_data, args.save_model)

