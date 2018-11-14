import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchnet as tnt
import numpy as np

def convert_to_torch_tensor(data_X, data_Y):
    
    X, Y = [torch.from_numpy(x) for x in data_X], torch.from_numpy(data_Y)
    return X, Y

loss_function = nn.BCELoss( reduction='sum')
    
def train(logging, model, weight, data_X, data_Y, validation_data_X, validation_data_Y, n_epochs, learning_rate, GPU, gpu_number, model_save_path, print_after_every = 2, validate_after_every = 2, save_after_every = 2):
    USE_CUDA = torch.cuda.device_count() >= 1 and GPU
    weight_tensor = torch.Tensor(weight).type(torch.FloatTensor)    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    data_X, data_Y = convert_to_torch_tensor(data_X, data_Y)

    if USE_CUDA:
        model = model.cuda(gpu_number)
        weight_tensor = weight_tensor.cuda(gpu_number)

    num_batches = data_X[0].shape[0]

    for epoch in range(n_epochs):
        confMatrix = tnt.meter.ConfusionMeter(5)
        for batch_idx in range(num_batches):
            X = [x[batch_idx] for x in data_X] 
            Y = torch.max(Variable(data_Y[batch_idx]), 1)[1]
            if USE_CUDA:
                X = [x.cuda(gpu_number) for x in X]
                Y = Y.cuda(gpu_number)
            optimizer.zero_grad()
            prediction = model(X)
            confMatrix.add(prediction.clone().detach(),Y.clone().detach())
            loss = F.cross_entropy(prediction, Y, weight = weight_tensor)
            loss.backward()
            optimizer.step()
            if batch_idx % print_after_every == 0:
                logging.info('Train Epoch: {} Batch Index: {} Total Number of Batches: {} \tLoss: {:.6f}'.format(
                    epoch, batch_idx , num_batches, loss.item()))

        logging.info('\nConfusion Matrix on Train for epoch {} \n {}\n'.format(epoch,
                        confMatrix.value()))
        if epoch % validate_after_every == 0:
            if validation_data_X[0].shape[0] * validation_data_X[0].shape[1] > 0:
                logging.info('Testing on Validation Set')
                test(logging, model, weight, validation_data_X, validation_data_Y, GPU, gpu_number)
            else:
                logging.info("Validation Set Size = 0, not validating")

        if epoch % save_after_every == 0:
            model_save_file = model_save_path + model.name() + "_" + str(epoch) + ".pth"
            logging.info("Saving model to {}".format(model_save_file))
            if USE_CUDA:
                torch.save(model.cpu().state_dict(), model_save_file)
                model = model.cuda(gpu_number)
            else:
                torch.save(model.state_dict(), model_save_file)
            logging.info("Model saved Successfully")

def test(logging, model, weight, data_X, data_Y, GPU, gpu_number):
    USE_CUDA = torch.cuda.device_count() >= 1 and GPU
    weight_tensor = torch.Tensor(weight).type(torch.FloatTensor)    
    model.eval()
    data_X, data_Y = convert_to_torch_tensor(data_X, data_Y)

    if USE_CUDA:
        model = model.cuda(gpu_number)
        weight_tensor = weight_tensor.cuda(gpu_number)

    loss= 0
    correct = 0
    confMatrix = tnt.meter.ConfusionMeter(5)
    num_batches = data_X[0].shape[0]
    batch_size = data_X[0].shape[1]
    for batch_idx in range(num_batches):
        X = [x[batch_idx] for x in data_X]
        Y = torch.max(Variable(data_Y[batch_idx]), 1)[1]
        if USE_CUDA:
            X = [x.cuda(gpu_number) for x in X]
            Y = Y.cuda(gpu_number)
        prediction = model(X)
        confMatrix.add(prediction.clone().detach(),Y.clone().detach())
        loss += F.cross_entropy(prediction, Y, weight = weight_tensor, reduction='sum').item()
        prediction = prediction.data.max(1, keepdim=True)[1]
        correct += prediction.eq(Y.data.view_as(prediction)).cpu().sum()
    data_len = num_batches * batch_size
    loss /= data_len
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        loss, correct, data_len,
        100.0 * float(correct) / data_len))
    logging.info('\nConfusion Matrix on Test {}\n'.format(
        confMatrix.value()))
        
def train_with_loader(logging, model, weight, train_loader, val_loader, n_epochs, learning_rate, GPU, gpu_number, model_save_path, print_after_every = 2, validate_after_every = 2, save_after_every = 2):
    USE_CUDA = torch.cuda.device_count() >= 1 and GPU
    weight_tensor = torch.Tensor(weight).type(torch.FloatTensor)    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    #data_X, data_Y = convert_to_torch_tensor(data_X, data_Y)

    if USE_CUDA:
        model = model.cuda(gpu_number)
        weight_tensor = weight_tensor.cuda(gpu_number)

    num_batches = len(train_loader)
    batch_size = train_loader.batch_size
    data_len = num_batches * batch_size

    for epoch in range(n_epochs):
        #confMatrix = tnt.meter.ConfusionMeter(5)
        correct = 0
        for batch_idx, (X, Y) in enumerate(train_loader):
            #rint("Training ", X, Y)
            #X, Y = convert_to_torch_tensor(X, Y)
            #X = [x[batch_idx] for x in data_X] 
            
            if USE_CUDA:
                X = [x.cuda(gpu_number) for x in X]
                Y = Y.cuda(gpu_number)
            optimizer.zero_grad()
            prediction = model(X)
            prediction = torch.sigmoid(prediction)
            #confMatrix.add(prediction.clone().detach(),Y.clone().detach())
            loss = loss_function(prediction, Y)
            loss.backward()
            optimizer.step()
            pred_caps = torch.zeros_like(prediction)
            pred_caps[prediction > 0.5] = 1
            correct += np.sum(np.all(pred_caps.cpu().numpy() == Y.cpu().numpy(), axis=1))
            if batch_idx % print_after_every == 0:
                logging.info('Train Epoch: {} Batch Index: {} Total Number of Batches: {} \tLoss: {:.6f}'.format(
                    epoch, batch_idx , num_batches, loss.item()))

        #logging.info('\nConfusion Matrix on Train for epoch {} \n {}\n'.format(epoch,
        #                confMatrix.value()))
        logging.info('\nTraining set: Accuracy: {}/{} ({:.4f}%)\n'.format(
            correct, data_len, 100.0 * float(correct) / data_len))
        if epoch % validate_after_every == 0:
            if len(val_loader) > 0:
                logging.info('Testing on Validation Set')
                test_with_loader(logging, model, weight, val_loader, GPU, gpu_number)
            else:
                logging.info("Validation Set Size = 0, not validating")

        if epoch % save_after_every == 0:
            model_save_file = model_save_path + model.name() + "_" + str(epoch) + ".pth"
            logging.info("Saving model to {}".format(model_save_file))
            if USE_CUDA:
                torch.save(model.cpu().state_dict(), model_save_file)
                model = model.cuda(gpu_number)
            else:
                torch.save(model.state_dict(), model_save_file)
            logging.info("Model saved Successfully")

def test_with_loader(logging, model, weight, test_loader, GPU, gpu_number):
    USE_CUDA = torch.cuda.device_count() >= 1 and GPU
    weight_tensor = torch.Tensor(weight).type(torch.FloatTensor)    
    model.eval()
    #data_X, data_Y = convert_to_torch_tensor(data_X, data_Y)

    if USE_CUDA:
        model = model.cuda(gpu_number)
        weight_tensor = weight_tensor.cuda(gpu_number)

    loss= 0
    correct = 0
    #confMatrix = tnt.meter.ConfusionMeter(5)
    num_batches = len(test_loader)
    batch_size = test_loader.batch_size
    for batch_idx, (X, Y) in enumerate(test_loader):
        #X = [x[batch_idx] for x in data_X]
        
        if USE_CUDA:
            X = [x.cuda(gpu_number) for x in X]
            Y = Y.cuda(gpu_number)
        prediction = model(X)
        prediction = torch.sigmoid(prediction)
        #confMatrix.add(prediction.clone().detach(),Y.clone().detach())
        loss += loss_function(prediction, Y).item()
        pred_caps = torch.zeros_like(prediction)
        pred_caps[prediction > 0.5] = 1
        correct += np.sum(np.all(pred_caps.cpu().numpy() == Y.cpu().numpy(), axis=1))

    data_len = num_batches * batch_size
    loss /= data_len
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        loss, correct, data_len,
        100.0 * float(correct) / data_len))
    #logging.info('\nConfusion Matrix on Test \n{}\n'.format(
     #   confMatrix.value()))




