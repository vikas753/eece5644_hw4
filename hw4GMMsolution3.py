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
import matplotlib.colors as colors
 
lossArray = []

## Create a 5-dim normalised feature vector and then reshape it to a 1d array
## with normalisation . 
    
num_dim = 5

image = imread('42049_color_res.jpg')

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

colors_l = ['#F0F8FF','#FAEBD7','#00FFFF','#7FFFD4','#F0FFFF','#F5F5DC','#FFE4C4', '#000000', '#FFEBCD','#0000FF', '#8A2BE2','#A52A2A', '#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50','#6495ED','#FFF8DC','#DC143C']

## PCA decomposition
for i in range(int(principalComponents.shape[0])):
    plt.scatter(principalComponents[i][0],principalComponents[i][1],c = 'r',s=50)
plt.show()

Data = np.array(principalComponents)
print(Data.shape)

## GMM decomposition into 2 components
gmm = GaussianMixture(n_components = 3,covariance_type='spherical',max_iter=1000,n_init=10)
gmm.fit(Data)
labels = gmm.predict(Data)

for i in range(int(principalComponents.shape[0])):
    plt.scatter(principalComponents[i][0],principalComponents[i][1],c = colors_l[labels[i]],s=50)
plt.show()

max_log_ll = 1
n_comp_opt = 1

## KFold validation goes here ::
for n_comp in range(20):
    n_comp = n_comp+1
    gmm = GaussianMixture(n_components = n_comp,covariance_type='spherical',max_iter=1000,n_init=10)
    gmm.fit(Data)
    labels = gmm.predict(Data)
    log_ll = gmm.lower_bound_
    print(" Log_LL = " , log_ll , " n_comp = " , n_comp)
    if(log_ll > max_log_ll):
        max_log_ll = log_ll
        n_comp_opt = n_comp    
    
print(" Log_LL_opt = " , max_log_ll , " n_comp_opt = " , n_comp_opt)     
gmm = GaussianMixture(n_components = n_comp_opt,covariance_type='spherical',max_iter=1000,n_init=10)
gmm.fit(Data)
labels = gmm.predict(Data)
for i in range(int(principalComponents.shape[0])):
    plt.scatter(principalComponents[i][0],principalComponents[i][1],c = colors_l[labels[i]],s=50)
    
plt.show()
            

    
