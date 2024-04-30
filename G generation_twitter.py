
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split

dataset = range(1164)
torch.manual_seed(42)
train_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[931, 233]
)

h_values = np.load('h_values3.npy',allow_pickle=True)
h_values = np.delete(h_values, 355, axis=0)
h_values = np.delete(h_values, 510, axis=0)
h_values = np.delete(h_values, 912, axis=0)
inputData = []
for i in range(931):
    inputData.append(h_values[train_dataset[i]])
for i in range(233):
    inputData.append(h_values[test_dataset[i]])

X = inputData
X_train = inputData[0:931]
print(np.shape(X))
print(np.shape(X_train))

G = []
for i in range(len(X)):
    print(i)
    x_j = X[i]
    G_j = np.full((1164,1164),0.0)
    for j in range(len(X)):
        for k in range(len(X)):
            G_j[j][k] = np.dot(x_j-X[j],np.transpose(x_j-X[k]))
    G.append(G_j)

np.save('G_twitter_random.npy',G)
