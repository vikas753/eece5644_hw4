import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
from sklearn.model_selection import LeavePOut
import scipy.stats as stats
import math

def toy_experiment(N):
    mu = 0 
    variance = 1    
    sigma = math.sqrt(variance)
    filedataX = np.linspace(mu - 3*sigma, mu + 3*sigma, N)
    filedataY = stats.norm.pdf(filedataX, mu, sigma)
    
    print(filedataX[0 10])
    
    plt.plot(filedataX, filedataY, ".")
    X , Y = Variable(torch.FloatTensor([filedataX]), requires_grad=True) , Variable(torch.FloatTensor([filedataY]), requires_grad=False) 
    
    OutX = X.detach().numpy()
    OutY = Y.detach().numpy()
    print(X)
    tf_X = Variable(torch.FloatTensor([OutX[0][0:5]]),requires_grad=True)
    print(tf_X)
    plt.plot(OutX,OutY,"*")
    plt.show()
    
toy_experiment(10)