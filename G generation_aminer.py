import pprint
from cvxopt import matrix, solvers
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split

dataset = range(10754)
torch.manual_seed(42)
train_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[8603, 2151]
)

adj_m = np.load('adj_m.npy',allow_pickle=True)
all_neighbour = np.load('all_neighbour.npy',allow_pickle=True)
h_values = np.load('h_values_mag_500.npy',allow_pickle=True)
inputData = []
for i in range(8603):
    inputData.append(h_values[train_dataset[i]])
for i in range(2151):
    inputData.append(h_values[test_dataset[i]])

X = inputData
X_train = inputData[0:8603]
print(np.shape(X))
print(np.shape(X_train))
G = []
for i in range(len(X)):
    print(i)
    x_j = X[i]
    G_j = np.full((150,150),0.0)
    for j in range(len(all_neighbour[i])):
        for k in range(len(all_neighbour[i])):
            G_j[j][k] = np.dot(x_j-X[all_neighbour[i][j]],np.transpose(x_j-X[all_neighbour[i][k]]))
    G.append(G_j)

np.save('G_mag.npy',G)
