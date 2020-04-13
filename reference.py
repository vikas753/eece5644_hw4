import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
## ========================================= Dataset Class =========================================== ##
class p1Dataset(Dataset):
    def __init__(self, csv_file, num_classes):
        xy = np.loadtxt(csv_file, delimiter=',', dtype=np.float32)
        self.n_samples = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 1:])  # size [n_samples, n_features]
        self.y_data = np.squeeze(xy[:, [0]])
        self.y_data = self.y_data - 1
        self.y_data = torch.from_numpy(self.y_data).long()

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

## =======================================  NeuralNet Class  ========================================== ##
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation = 'sigmoid'):   # sigmoid/elu
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.sig = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.activation = activation

    def forward(self, x):
        out = self.l1(x)
        if self.activation is 'elu':
            out = self.elu(out)
        elif self.activation is 'sigmoid':
            out = self.sig(out)
        out = self.l2(out)
        return out

## ===================================== K-Fold ================================================== ##
def kfold( K, model, dataset, verbose = False):
    #K = 10
    n_samples = len(dataset)
    batch_size = int(n_samples / K)
    avg_acc = 0
    for k in range(1, K+1):
        if verbose:
            print('-----------------------------------------------------------------')
            print('K = ', k)
        features_val, labels_val = dataset[(k-1)*batch_size : k*batch_size]
        if k == 1:
            features_train, labels_train = dataset[k*batch_size:]
        elif k == K:
            features_train, labels_train = dataset[:(k-1)*batch_size]
        else:
            features_train1, labels_train1 = dataset[:(k - 1) * batch_size]
            features_train2, labels_train2 = dataset[k * batch_size:]
            features_train = torch.cat([features_train1, features_train2], dim=0)
            labels_train = torch.cat([labels_train1, labels_train2], dim=0)

        #moving data to selected device
        features_val = features_val.to(device)
        labels_val = labels_val.to(device)
        features_train = features_train.to(device)
        labels_train = labels_train.to(device)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(features_train)
            loss = criterion(outputs, labels_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % (num_epochs/10) == 0:
                print('Epoch ', (epoch + 1), '/', num_epochs, ' Loss: ', loss.item())
        '''
        # Print weights
        for name, param in model.named_parameters():
            print(name, param.data)
         '''
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            outputs = model(features_val)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels_val.size(0)
            n_correct += (predicted == labels_val).sum().item()

        acc = 100.0 * n_correct / n_samples
        avg_acc += acc
        if verbose:
            print('Accuracy: ', acc)

    avg_acc = avg_acc / K
    if verbose:
        print('Average Accuracy: ', avg_acc)
    return avg_acc

## ====================================== Plotting function =========================================== ##
def show_plot(sigmoid_array, elu_array, dataset):
    x = np.arange(len(elu_array))  # the label locations
    width = 0.35  # the width of the bars
    labels = [str(n+1) for n in x]
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, sigmoid_array, width, label='sigmoid activation fn')
    rects2 = ax.bar(x + width / 2, elu_array, width, label='elu activation fn')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('(1 - P(error))*100')
    ax.set_xlabel('No. of Perceptrons In First Layer')
    ax.set_title('Accuracy on ' + dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()

## =========================== choosing a model from performace measure from kfold  ================================= ##
def select_a_model(sigmoid_array, elu_array):
    max_acc_sigmoid = max(sigmoid_array)
    max_acc_sigmoid_perceptron = np.argmax(sigmoid_array) + 1
    max_acc_elu = max(elu_array)
    max_acc_elu_perceptron = np.argmax(elu_array) + 1
    max_acc_activation = np.argmax([max_acc_sigmoid, max_acc_elu])
    if max_acc_activation == 0:
        return 'sigmoid', max_acc_sigmoid_perceptron
    elif max_acc_activation == 1:
        return 'elu', max_acc_elu_perceptron

## ================================= Model Order Selection Using D100_train ======================================= ##
device = 'cpu'
D100_train_dataset = p1Dataset('csv_data/D100_train.csv', 3)
sigmoid_array = []
elu_array = []
no_of_perceptron = 12
for act in ['sigmoid', 'elu']:
    for nPerceptron in range(0, no_of_perceptron):
        net = NeuralNet(2, nPerceptron+1, 3, activation=act).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        num_epochs = 1000
        p = kfold(10, net, D100_train_dataset, verbose=False)
        if act == 'sigmoid':
            sigmoid_array.append(p)
        elif act == 'elu':
            elu_array.append(p)
        print(act, nPerceptron+1, p)
show_plot(sigmoid_array, elu_array, 'D100_train')
selected_model_D100 = select_a_model(sigmoid_array, elu_array)
print('## ============= Model Order Selection Result for D100_train =============== ##')
print('Competing Models: [sigmoid activation, elu activation] x '+str(no_of_perceptron)+' perceptrons')
print('Selected Activation fn: ', selected_model_D100[0])
print('Selected No. of Perceptrons: ', selected_model_D100[1])
print('## ========================================================================= ##')

## ============================= Training the selected model and Evaluating with Dtest ============================= ##
net_D100 = NeuralNet(2, selected_model_D100[1], 3, activation=selected_model_D100[0]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net_D100.parameters(), lr=0.1)
num_epochs = 1500
features, labels = D100_train_dataset[:]
features = features.to(device)
labels = labels.to(device)

Dtest_dataset = p1Dataset('csv_data/D10000_test.csv', 3)
features_test, labels_test = Dtest_dataset[:]
features_test = features_test.to(device)
labels_test = labels_test.to(device)

for epoch in range(num_epochs):
    # Forward pass
    outputs = net_D100(features)
    loss = criterion(outputs, labels)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    outputs = net_D100(features_test)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    n_samples += labels_test.size(0)
    n_correct += (predicted == labels_test).sum().item()
    D100_acc = 100.0 * n_correct / n_samples
    print('Accuracy: ', D100_acc)
print('## ========================================================================= ##')
## ================================= Model Order Selection Using D500_train ======================================= ##
D500_train_dataset = p1Dataset('csv_data/D500_train.csv', 3)
sigmoid_array = []
elu_array = []
no_of_perceptron = 12
for act in ['sigmoid', 'elu']:
    for nPerceptron in range(0, no_of_perceptron):
        net = NeuralNet(2, nPerceptron+1, 3, activation=act).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        num_epochs = 1000
        p = kfold(10, net, D500_train_dataset, verbose=False)
        if act == 'sigmoid':
            sigmoid_array.append(p)
        elif act == 'elu':
            elu_array.append(p)
        print(act, nPerceptron+1, p)
show_plot(sigmoid_array, elu_array, 'D500_train')
selected_model_D500 = select_a_model(sigmoid_array, elu_array)
print('## ============= Model Order Selection Result for D500_train =============== ##')
print('Competing Models: [sigmoid activation, elu activation] x '+str(no_of_perceptron)+' perceptrons')
print('Selected Activation fn: ', selected_model_D500[0])
print('Selected No. of Perceptrons: ', selected_model_D500[1])
print('## ========================================================================= ##')

## ============================= Training the selected model and Evaluating with Dtest ============================= ##
net_D500 = NeuralNet(2, selected_model_D500[1], 3, activation=selected_model_D500[0]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net_D500.parameters(), lr=0.1)
num_epochs = 1000
features, labels = D500_train_dataset[:]
features = features.to(device)
labels = labels.to(device)

for epoch in range(num_epochs):
    # Forward pass
    outputs = net_D500(features)
    loss = criterion(outputs, labels)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    outputs = net_D500(features_test)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    n_samples += labels_test.size(0)
    n_correct += (predicted == labels_test).sum().item()
    D500_acc = 100.0 * n_correct / n_samples
    print('Accuracy: ', D500_acc)
print('## ========================================================================= ##')
## ================================= Model Order Selection Using D1000_train ======================================= ##
D1000_train_dataset = p1Dataset('csv_data/D1000_train.csv', 3)
sigmoid_array = []
elu_array = []
no_of_perceptron = 12
for act in ['sigmoid', 'elu']:
    for nPerceptron in range(0, no_of_perceptron):
        net = NeuralNet(2, nPerceptron+1, 3, activation=act).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        num_epochs = 1000
        p = kfold(10, net, D1000_train_dataset, verbose=False)
        if act == 'sigmoid':
            sigmoid_array.append(p)
        elif act == 'elu':
            elu_array.append(p)
        print(act, nPerceptron+1, p)
show_plot(sigmoid_array, elu_array, 'D1000_train')
selected_model_D1000 = select_a_model(sigmoid_array, elu_array)
print('## ============= Model Order Selection Result for D1000_train =============== ##')
print('Competing Models: [sigmoid activation, elu activation] x '+str(no_of_perceptron)+' perceptrons')
print('Selected Activation fn: ', selected_model_D1000[0])
print('Selected No. of Perceptrons: ', selected_model_D1000[1])
print('## ========================================================================= ##')

## ============================= Training the selected model and Evaluating with Dtest ============================= ##
net_D1000 = NeuralNet(2, selected_model_D1000[1], 3, activation=selected_model_D1000[0]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net_D1000.parameters(), lr=0.1)
num_epochs = 1500
features, labels = D1000_train_dataset[:]
features = features.to(device)
labels = labels.to(device)

for epoch in range(num_epochs):
    # Forward pass
    outputs = net_D1000(features)
    loss = criterion(outputs, labels)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    outputs = net_D1000(features_test)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    n_samples += labels_test.size(0)
    n_correct += (predicted == labels_test).sum().item()
    D1000_acc = 100.0 * n_correct / n_samples
    print('Accuracy: ', D1000_acc)
print('## ========================================================================= ##')

## ============================= D100 vs D500 vs D1000 ============================ ##
plt.figure()
plt.bar(['model_D100', 'model_D500', 'model_D1000'], [D100_acc, D500_acc, D1000_acc])
plt.title('Accuracy of models on Dtest')
plt.ylabel('Accuracy')

## ====================================== Display Plots ===================================== ##
