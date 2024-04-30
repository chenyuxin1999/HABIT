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
h_values = np.load('h_values_aminer_400.npy',allow_pickle=True)
inputData = []
for i in range(8603):
    inputData.append(h_values[train_dataset[i]])
for i in range(2151):
    inputData.append(h_values[test_dataset[i]])

X = inputData
X_train = inputData[0:8603]
print(np.shape(X))
print(np.shape(X_train))

W = []
for i in range(len(X)):
    print(i)
    x_j = X[i]
    G_j = np.full((10754,10754),0.0)
    x_j_expand = np.expand_dims(x_j, axis=0)
    x_j_repeat = np.repeat(x_j_expand, 10754, axis=0)
    G_j = np.matmul(x_j_repeat - X, (x_j_repeat - X).T)
    print(G_j)

    zeroArray = [0 for i in range(10754)]
    oneArray = [1.0 for i in range(10754)]
    gaa = G_j * np.array(adj_m[i]).reshape(10754, 1) * np.transpose(np.array(adj_m[i]).reshape(10754, 1))
    a_value = np.multiply(oneArray, adj_m[i])

    P = matrix(gaa.astype('double'))
    q = matrix(np.array(zeroArray).reshape(-1, 1).astype('double'))
    G = matrix(np.identity(10754).astype('double') * (-1))
    h = matrix(np.array(zeroArray).reshape(-1, 1).astype('double'))
    A = matrix(np.array(a_value).reshape(1, 10754).astype('double'))
    b = matrix([1.0])
    result = solvers.qp(P, q, G, h, A, b)

    print(np.array(result['x']))

    w_j = np.multiply(np.array(result['x']), adj_m[i].reshape(10754, 1))
    W.append(w_j)

W_reshape = np.array(W).reshape(10754,10754)
W_reshape = np.transpose(W_reshape)
W = W_reshape
np.save('W_aminer.npy',W)
