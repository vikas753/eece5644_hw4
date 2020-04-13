import matplotlib.pyplot as plt
import numpy as np
import csv
import random
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold
import scipy.stats as stats
import math
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

## Generates a gaussian distribution or a toy data set for 
## verification of a  working of NN
def normal(x,mu,sigma):
    return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )

colors = ['red','green','blue','purple']

def show_plot_svm_contour(data,svm,filename,labels):
    dataX = []
    dataY = []
    for i in range(len(data)):
        dataX.append(data[i][0])
        dataY.append(data[i][1])

    dataX = np.array(dataX)
    dataY = np.array(dataY)
        
    x_min, x_max = dataX.min() - 1, dataX.max() + 1
    y_min, y_max = dataY.min() - 1, dataY.max() + 1
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.scatter(dataX,dataY,c=labels,cmap=plt.cm.coolwarm)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    if filename == "":
        plt.show()
    else:
        plt.savefig(filename)

def plot_scatter_sh(data,labels,filename):
    dataX = []
    dataY = []
    for i in range(len(data)):
        dataXY=data[i]
        dataX.append(dataXY[0])
        dataY.append(dataXY[1])

    plt.scatter(dataX,dataY,c=labels,cmap=plt.cm.coolwarm)
    if filename == "":
        plt.show()
    else:
        plt.savefig(filename)
        

## Trains a SVM  with C and gamma . 
def svmtraining(regularization,gamma,data,labels):
    model = SVC(C=regularization,gamma=gamma,kernel='rbf')
    model.fit(data, labels)
    #plot_scatter_sh(data,labels,"")
    #plot_scatter_sh(data,predicted_labels,"")
    return model    
    
## Performs a validation of a SVM against a data set and returns score
def svmvalidate(svm,data,labels,filename):
    #plot_scatter_sh(data,labels,filename)
    predicted_labels=svm.predict(data)
    loss=accuracy_score(labels,predicted_labels)
    print("loss=",loss)
    #show_plot_svm_contour(data,svm,filename,labels)
    return loss
    
## Performs a full training on the data using SVM .     
def svmFullTraining(data,labels):
    min_loss = 1000
    C_opt = 0 
    Gamma_opt = 0
    for C in [1,10,100,200,500,700,1000]:
        for Gamma in [0.00001,0.0001,0.001,0.01,0.1,1,10]:
            model=svmtraining(C,Gamma,data,labels)
            loss=svmvalidate(model,data,labels,"")
            if(loss < min_loss):
                min_loss = loss
                best_model = model
                C_opt = C
                Gamma_opt = Gamma
    print(" C_opt = " , C_opt , " Gamma_opt = " , Gamma_opt)
    return best_model                    

## Performs a K-fold cross validation using LeavePOut SkLearn module
def KfoldCrossValidation(data,labels,kfold):
    dataLength = len(data) 
    batchSize = int(dataLength / kfold)
    lpo_cross_val_model = KFold(n_splits=kfold)
    data_np = np.array(data)

    print("batchSize =" , batchSize) 
    min_loss = 10000
    validationIndexArray = []
    lossKfoldArray = []
    validationIndex = 0
    
    for train_index, test_index in lpo_cross_val_model.split(data_np):
        validationIndexArray.append(validationIndex)
        validationIndex = validationIndex + 1
        
        data_train = data_np[train_index]
        
        data_test  = data_np[test_index]
        labels_train = labels[train_index]
        labels_test  = labels[test_index]
        trainedSVM = svmFullTraining(data_train,labels_train)
        loss_dataset = svmvalidate(trainedSVM,data_test,labels_test,"modelTestExample.png")
        lossKfoldArray.append(loss_dataset)
        
        if(loss_dataset < min_loss):
            min_loss = loss_dataset
            svmmodel = trainedSVM
    
    plt.plot(validationIndexArray,lossKfoldArray, linewidth=5.0)
    plt.ylabel(' Loss-K-fold Array ')
    plt.xlabel(' Validation Index ')
    plt.savefig('ValidationIndexGraph.png')
     
    return svmmodel    
 
lossArray = []
    
with open('dataXhw421000train.csv', newline='') as f:
    reader = csv.reader(f)
    filedataX_train = [tuple(row) for row in reader]

with open('dataYhw421000train.csv', newline='') as f:
    reader = csv.reader(f)
    filedataY_train = [tuple(row) for row in reader]

with open('labelshw421k.csv', newline='') as f:
    reader = csv.reader(f)
    labels_train = [tuple(row) for row in reader]

with open('dataXhw4110000test.csv', newline='') as f:
    reader = csv.reader(f)
    filedataX_test = [tuple(row) for row in reader]

with open('dataYhw4110000test.csv', newline='') as f:
    reader = csv.reader(f)
    filedataY_test = [tuple(row) for row in reader]

with open('labelshw4210k.csv', newline='') as f:
    reader = csv.reader(f)
    labels_test = [tuple(row) for row in reader]

filedataX_train = [float(i) for i in filedataX_train[0]]    
filedataY_train = [float(j) for j in filedataY_train[0]]
labels_train = [int(k)   for k in labels_train[0]]

filedataX_train = np.array(filedataX_train)
filedataY_train = np.array(filedataY_train)
labels_train = np.array(labels_train)

filedataX_test = [float(i) for i in filedataX_test[0]]    
filedataY_test = [float(j) for j in filedataY_test[0]]
labels_test = [int(k)   for k in labels_test[0]]

filedataX_test = np.array(filedataX_test)
filedataY_test = np.array(filedataY_test)
labels_test = np.array(labels_test)

    
Data = []

for i in range(filedataX_train.size):
    Data.append([filedataX_train[i],filedataY_train[i]])

Data_test = []

for i in range(filedataX_test.size):
    Data_test.append([filedataX_test[i],filedataY_test[i]])
        
svmmodel = KfoldCrossValidation(Data,labels_train,10)   
loss = svmvalidate(svmmodel,Data_test,labels_test,"")
print(" Loss of SVM = " , loss)