




# Input that may vary
batch_size = 160


# Libraries
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data
# The custom-defined model
from QOL_Library.ConvTimeLSTM1 import ConvTime_LSTM1
# Function to separate whole sequences into prior and current scenes
from QOL_Library.Separate_X_Y import sep_x_y
# Formal PyTorch data set classes that increase hardware utilization
from QOL_Library.Dataset_Classes import train_Dataset, validation_Dataset
import nvidia_smi


# Marking the begin time
print(datetime.datetime.now())

print("importing data")

# Import Moving MNIST
Moving_MNIST = np.load('data/mnist_test_seq.npy')
Moving_MNIST = Moving_MNIST / 255
Moving_MNIST.shape


# Give PyTorch the data
# Making into PyTorch tensor
Moving_MNIST_tensor = torch.from_numpy(Moving_MNIST)
# Putting the existing dimensions into appropriate order
Moving_MNIST_tensor = Moving_MNIST_tensor.permute(1, 0, 2, 3)
# Added the acknowledge that this is 1 spectral band
Moving_MNIST_tensor = Moving_MNIST_tensor.unsqueeze(2)
# Checking shape
Moving_MNIST_tensor.shape

print("processing data")

# Train/validation split
train_indices = np.random.choice(range(10000), size = 8000, replace = False)
OutofSample_indices = [index for index in range(10000) if index not in train_indices.tolist()]
validation_indices = np.random.choice(OutofSample_indices, size = 1000, replace = False)


# Separate x (previous 10 in seq) and y (next in seq)
x, y = sep_x_y(Moving_MNIST_tensor[train_indices])
x_validation, y_validation = sep_x_y(Moving_MNIST_tensor[validation_indices])

print("setting up the model")

# Picking one of the like-sequence tensors within the list to set parameters
channels = x.shape[2]
height = x.shape[3]
width = x.shape[4]


# Set model hyperparameters
conv_time_lstm = ConvTime_LSTM1(input_size = (height,
                                              width),
                                input_dim = channels,
                                hidden_dim = [128, 64, 64, 1],
                                kernel_size = (5, 5),
                                num_layers = 4,
                                batch_first = True,
                                bias = True,
                                return_all_layers = False,
                                GPU = True)
# Give it to the GPU
conv_time_lstm.cuda()


# Training
# Optimization methods
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(conv_time_lstm.parameters())

print("facilitating parallel operations")

# Pass our data to those classes
training_set = train_Dataset(x,
                             y,
                             data_indices=range(y.shape[0]))
validation_set = validation_Dataset(x_validation,
                                    y_validation,
                                    data_indices=range(y_validation.shape[0]))
train_loader = torch.utils.data.DataLoader(dataset = training_set,
                                           batch_size = batch_size,
                                           shuffle = True)
validation_loader = torch.utils.data.DataLoader(dataset = validation_set,
                                                batch_size = batch_size,
                                                shuffle = True)


# Pass formal device argument, preferring GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Allow parallelization
conv_time_lstm = torch.nn.DataParallel(conv_time_lstm)

print("training loop")
print(datetime.datetime.now())

# Training loop
loss_list = []
epochs = int(np.ceil((7*10**5) / x.shape[0]))
for i in range(epochs):
    for data in train_loader:
        # data loader
        batch_x, batch_y = data
        # move to GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # run model and get the prediction
        batch_y_hat = conv_time_lstm(batch_x,
                                     torch.ones_like(batch_x))
        batch_y_hat = batch_y_hat[0][0][:, -2:-1, :, :, :]
        # calculate and store the loss
        batch_loss = loss(batch_y, batch_y_hat)
        loss_list.append(batch_loss.item())
        # update parameters
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    print('Epoch: ', i, '\n\tBatch loss: ', batch_loss.item(), '\n')
    # Printing gpu perf
    nvidia_smi.nvmlInit()
    for gpu_core in range(4):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_core)
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')


# Marking the end time
print("end of training loop")
print(datetime.datetime.now())




