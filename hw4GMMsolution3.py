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
from skimage.io import imread, imshow
from sklearn.mixture import GaussianMixture 
from sklearn.decomposition import PCA
from skimage.transform import rescale, resize, downscale_local_mean


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

## Create a 5-dim normalised feature vector and then reshape it to a 1d array
## with normalisation . 
    
num_dim = 5

image = imread('3096_color.jpg')

feature_matrix = np.zeros((image.shape[0]*image.shape[1],num_dim))
print("fm_shape=" , image.shape)

for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        feature_matrix[i*j][0] = i / image.shape[0]
        feature_matrix[i*j][1] = j / image.shape[1]
        feature_matrix[i*j][2] = image[i,j,0] / 256
        feature_matrix[i*j][3] = image[i,j,1] / 256
        feature_matrix[i*j][4] = image[i,j,2] / 256
        
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(feature_matrix)

print(principalComponents.shape)
for i in range(int(principalComponents.shape[0]/100)):
    plt.scatter(principalComponents[i][0],principalComponents[i][1],c = 'r',s=50)
plt.show()

Data = np.array(principalComponents[0:int(principalComponents.shape[0]/100),:])
print(Data.shape)
gmm = GaussianMixture(n_components = 3,covariance_type='spherical',max_iter=1000,n_init=10)
gmm.fit(Data)
labels = gmm.predict(Data)
log_ll = gmm.lower_bound_
print(" Log_LL = " , log_ll)

for i in range(int(Data.shape[0])):
    if(labels[i] == 0):
        plt.scatter(Data[i][0],Data[i][1], c ='r') 
    else:
        plt.scatter(Data[i][0],Data[i][1], c ='yellow') 

plt.show()
