from main_model import MainModel
import torch
import numpy as np
import logging
from GTA_data_set_sequential import GTADataSetSequential
from find_stats import findStats
import json
from datetime import datetime
import sys

with open("configs/config_test.json",'r') as file:
    config = json.load(file)

logging.basicConfig(level=logging.INFO,
    filename= config['log_dir'] + config['log_name'] + 'last_layer_' + datetime.now().strftime('%d_%m_%Y_%H_%M_%S.log'),
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


if config['print']:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(ch)




weights = np.array([0.001, 1.0, 1.0, 1.0, 1.0])

# 8.1e-3, 1, 1, 0.1, 0.1

stats = findStats(logging, None, False)
val_dataset = GTADataSetSequential(config['val_data'], stats, logging)
logging.info("Number of Validation Examples : {}".format(len(val_dataset)))

val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = config['batch_size'], shuffle = False, num_workers = 0)

logging.info("Creating Model Graph")
model = MainModel(config['model_dict'],5)
model.eval()
logging.info("Model Created successfully")

if len(sys.argv) < 2:
    logging.info("Please provide the path to load model !!")
    sys.exit()
logging.info("Loading Model Weights from {}".format(sys.argv[1]))
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
logging.info("Model weights loaded successfully.")

epochs = 400

for epoch in range(epochs):
	correct = 0
	counts = np.array([0, 0, 0, 0, 0])
	USE_CUDA = torch.cuda.device_count() >= 1 and config['gpu']
	if USE_CUDA:
		model = model.cuda(config['gpu_number'])
	with torch.set_grad_enabled(False):
	    for batch_idx, (X, Y) in enumerate(val_loader):

	        Y = torch.max(Y, 1)[1]
	        if USE_CUDA:
	            X = [x.cuda(config['gpu_number']) for x in X]
	            Y = Y.cuda(config['gpu_number'])
	        prediction = model(X)
	        prediction = prediction.cpu()
	        prediction = torch.nn.Softmax()(prediction)

	        for idx, p in enumerate(prediction):
	        	p = p.numpy()
	        	p = p * weights
	        	p = np.argmax(p)
	        	counts[p] += 1
	        	if p == Y.data[idx]:
	        		correct += 1
	        logging.info("Epoch: {} Batch : {}/{}, Prediction counts : {}, correct : {}".format(epoch, batch_idx, len(val_loader), counts, correct))

	logging.info("Prediction counts : ",counts)
	logging.info("Correct: {}/{}".format(correct, len(val_loader) * config['batch_size']))
	max_class = np.argmax(counts)
	logging.info("Max class = ", max_class)
	weights[max_class] *= 0.9
	logging.info("Weights after epoch {} --> {}".format(epoch, weights))





