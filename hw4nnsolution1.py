import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
from sklearn.model_selection import LeavePOut,KFold
import scipy.stats as stats
import math

## Neural Network Class 
class Net(nn.Module):
    def __init__(self,num_perceptrons,num_inputs):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.fc1 = nn.Linear(num_inputs,num_perceptrons)
        self.fc2 = nn.Linear(num_perceptrons,num_inputs)
    def forward(self, x):
        outNN = self.fc2(F.relu(self.fc1(x)))
        return outNN
    def get_inputs(self):
        return self.num_inputs


## Generates a gaussian distribution or a toy data set for 
## verification of a  working of NN
def normal(x,mu,sigma):
    return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )

## Trains a neural network 
def nntraining(num_inputs,num_perceptrons,x,y):
    
    criterion = nn.MSELoss()
    net = Net(num_perceptrons,num_inputs)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.5)
    range_index = int(x.size/num_inputs)    
    
    epochLossArray = []
    epochArrayIndex = []
    
    for epoch in range(10000):
        avg_loss = 0
        OutY = np.array([])
        for start_index in range(range_index):
            end_idx = start_index+num_inputs
            X, Y = Variable(torch.FloatTensor([x[start_index:end_idx]]), requires_grad=True), Variable(torch.FloatTensor([y[start_index:end_idx]]), requires_grad=False)        
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            outputs_np_arr = outputs.detach().numpy()
            #print(outputs_np_arr)
            OutY = np.concatenate([OutY,outputs_np_arr[0]])
        avg_loss = avg_loss / range_index    
        epochLossArray.append(avg_loss)
        epochArrayIndex.append(epoch)        
        
        ##if epoch % 100 == 0:            
            ##print("Epoch {} - loss: {}".format(epoch, avg_loss))

    return net

## Performs a validation of a neural network against a data set 
def nnvalidate(nnet,dataX,dataY,filename):
    num_inputs = nnet.get_inputs()
    dataLength = dataX.size
    range_index = int(dataLength/num_inputs)    
    criterion = nn.MSELoss()
    
    if(filename == ""):
        plt.scatter(dataX,dataY,c="0")
    
    with torch.no_grad():
        avg_loss = 0
        OutY = np.array([])
        for start_index in range(range_index):
            end_idx = start_index+num_inputs
            X, Y = Variable(torch.FloatTensor([dataX[start_index:end_idx]]), requires_grad=True), Variable(torch.FloatTensor([dataY[start_index:end_idx]]), requires_grad=False)        
            outputs = nnet(X)
            loss = criterion(outputs, Y)
            avg_loss = avg_loss + loss.item()
            outputs_np_arr = outputs.detach().numpy()
            OutY = np.concatenate([OutY,outputs_np_arr[0]])
            avg_loss = avg_loss / range_index    
              
    if(filename == ""):
        plt.scatter(dataX,OutY,c="1")
        plt.show()
        plt.plot(dataY,OutY)
        plt.show()
    
    return avg_loss

## Performs a K-fold cross validation using LeavePOut SkLearn module
def KfoldCrossValidation(dataX,dataY,kfold,num_perceptrons):
    dataLength = dataX.size 
    batchSize = int(dataLength / kfold)
    lpo_cross_val_model = KFold(n_splits=kfold)
    min_loss = 10000
    validationIndexArray = []
    lossKfoldArray = []
    validationIndex = 0
    
    for train_index, test_index in lpo_cross_val_model.split(dataX):
        validationIndexArray.append(validationIndex)
        validationIndex = validationIndex + 1        
        X_train = dataX[train_index] 
        X_test = dataX[test_index]
        Y_train = dataY[train_index] 
        Y_test = dataY[test_index]
        
        trainedNN = nntraining(batchSize,num_perceptrons,X_train,Y_train)
        loss_dataset = nnvalidate(trainedNN,X_test,Y_test,"modelTestExample.png")
        lossKfoldArray.append(loss_dataset)
        
        if(loss_dataset < min_loss):
            min_loss = loss_dataset
            nnmodel = trainedNN
    
     
    return nnmodel     
 
 ## Main experiment runs here with dataset train and test samples
def Experiment(num_perceptrons,kfolds,filedataX_test,filedataY_test,filedataX_train,filedataY_train): 
    trainedNN = KfoldCrossValidation(filedataX_train,filedataY_train,kfolds,num_perceptrons)
    loss = nnvalidate(trainedNN,filedataX_train,filedataY_train,"ResultModel.png")
    return loss
    
lossArray = []
num_perceptrons = []
    
    
## Runs the routine repeatedly    
for i in range(20):
    
    with open('dataXhw411000train.csv', newline='') as f:
        reader = csv.reader(f)
        filedataX_train = [tuple(row) for row in reader]

    with open('dataYhw411000train.csv', newline='') as f:
        reader = csv.reader(f)
        filedataY_train = [tuple(row) for row in reader]

    with open('dataXhw4110000test.csv', newline='') as f:
        reader = csv.reader(f)
        filedataX_test = [tuple(row) for row in reader]

    with open('dataYhw4110000test.csv', newline='') as f:
        reader = csv.reader(f)
        filedataY_test = [tuple(row) for row in reader]
    
        
    filedataX_train = [float(i) for i in filedataX_train[0]]    
    filedataY_train = [float(j) for j in filedataY_train[0]]

    filedataX_test = [float(i) for i in filedataX_test[0]]    
    filedataY_test = [float(j) for j in filedataY_test[0]]
    
    
    filedataX_train = np.array(filedataX_train)
    filedataY_train = np.array(filedataY_train)
    
    filedataX_test = np.array(filedataX_test)
    filedataY_test = np.array(filedataY_test)
 
    print(filedataX_test)
    print(filedataY_test)
 
    num_perceptrons = i+1
    trainedNN = KfoldCrossValidation(filedataX_train,filedataY_train,10,num_perceptrons)
    loss = nnvalidate(trainedNN,filedataX_test,filedataY_test,"")
    
    print("num_perceptrons=",20)
    print("Loss=",loss)
    

