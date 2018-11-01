import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

def convert_to_torch_tensor(data_X, data_Y):
    
    X, Y = [torch.from_numpy(x) for x in data_X], torch.from_numpy(data_Y)
    return X, Y

def train(logging, model, data_X, data_Y, validation_data_X, validation_data_Y, n_epochs, learning_rate, GPU, gpu_number, model_save_path, print_after_every = 2, validate_after_every = 2, save_after_every = 2):
    USE_CUDA = torch.cuda.device_count() >= 1 and GPU
            
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    data_X, data_Y = convert_to_torch_tensor(data_X, data_Y)

    if USE_CUDA:
        model = model.cuda(gpu_number)

    num_batches = data_X[0].shape[0]

    for epoch in range(n_epochs):
        for batch_idx in range(num_batches):
            X = [x[batch_idx] for x in data_X] 
            Y = torch.max(Variable(data_Y[batch_idx]), 1)[1]
            if USE_CUDA:
                X = [x.cuda(gpu_number) for x in X]
                Y = Y.cuda(gpu_number)
            optimizer.zero_grad()
            prediction = model(X)
            loss = F.cross_entropy(prediction, Y)
            loss.backward()
            optimizer.step()
            if batch_idx % print_after_every == 0:
                logging.info('Train Epoch: {} Batch Index: {} Total Number of Batches: {} \tLoss: {:.6f}'.format(
                    epoch, batch_idx , data_X.shape[0], loss.item()))

        if epoch % validate_after_every == 0:
            if epoch % validate_after_every == 0:
                logging.info('Testing on Validation Set')
                test(logging, model, validation_data_X, validation_data_Y, GPU, gpu_number)
            else:
                logging.info("Validation Set Size = 0, not validating")

        if epoch % save_after_every == 0:
            logging.info("Saving model to {}".format(model_save_path))
            if USE_CUDA:
                torch.save(model.cpu().state_dict(), model_save_path)
                model = model.cuda(gpu_number)
            else:
                torch.save(model.state_dict(), model_save_path)
            logging.info("Model saved Successfully")

def test(logging, model, data_X, data_Y, GPU, gpu_number):
    USE_CUDA = torch.cuda.device_count() >= 1 and GPU
    model.eval()
    data_X, data_Y = convert_to_torch_tensor(data_X, data_Y)

    if USE_CUDA:
        model = model.cuda(gpu_number)

    loss= 0
    correct = 0
    num_batches = data_X[0].shape[0]
    batch_size = data_X[0].shape[1]
    for batch_idx in range(num_batches):
        X = [x[batch_idx] for x in data_X]
        Y = torch.max(Variable(data_Y[batch_idx]), 1)[1]
        if USE_CUDA:
            X = [x.cuda(gpu_number) for x in X]
            Y = Y.cuda(gpu_number)
        prediction = model(X)
        loss += F.cross_entropy(prediction, Y, reduction='sum').item()
        prediction = prediction.data.max(1, keepdim=True)[1]
        correct += prediction.eq(Y.data.view_as(prediction)).cpu().sum()
    data_len = num_batches * batch_size
    loss /= data_len
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        loss, correct, data_len,
        100.0 * float(correct) / data_len))




