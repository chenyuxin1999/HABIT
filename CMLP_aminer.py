import numpy as np
import torch.utils
from torch.utils.data import random_split
from scipy.optimize import linprog
from scipy import optimize
from sklearn.linear_model import Ridge
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

dataset = range(10754)
torch.manual_seed(42)
train_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[8603, 2151]
)

y = torch.load('label_OAG.pt')
y = torch.where(y > 0, 1, -1)
y = y.numpy()

Y_train_random = np.full((8603,419),0)
for i in range(8603):
    Y_train_random[i] = y[list(train_dataset)[i]]

Y_test_random = np.full((2151,419),0)
for i in range(2151):
    Y_test_random[i] = y[list(test_dataset)[i]]

Y_all = np.full((10754,419),0)
for i in range(8603):
    Y_all[i] = Y_train_random[i]

h_values = np.load('h_values_aminer_400.npy',allow_pickle=True)

inputData = []
for i in range(8603):
    inputData.append(h_values[train_dataset[i]])
for i in range(2151):
    inputData.append(h_values[test_dataset[i]])

X = inputData
X_train = inputData[0:8603]

g = np.load('G_aminer.npy',allow_pickle=True)
W = np.load('W_aminer.npy',allow_pickle=True)
D = np.load('D_aminer.npy',allow_pickle=True)

#P = D*W*D
P = np.dot(D,W)
P = np.dot(P,D)

F_0 = Y_all
Z_0 = Y_train_random
I = np.mat(np.identity(10754))

Z_all = np.full((10754,419),0.0)
for i in range(8603):
    Z_all[i] = Z_0[i]

n_samples, n_features = 8603, 419
np.random.seed(0)
y = -0.1*Y_train_random[:,0]
X = -0.1*Y_train_random
clf = Ridge(alpha=1.0)
clf.fit(X, y)
clf = Ridge(alpha=1.0)
r = []
for i in range(419):
    y = -0.1*Y_train_random[:,i]
    X = -0.1*Y_train_random
    clf.fit(X, y)
    r.append(clf.coef_)
r = np.transpose(r)
np.shape(r)

Y_u = Y_test_random
for i in range(len(Y_u)):
    for j in range(len(Y_u[i])):
        if Y_u[i][j] == -1:
            Y_u[i][j] = 0

writer = SummaryWriter('./CMLP')

B = 0.1
u = 0.1
v = 0.1
g = v / u
a = 0.1

I_1164 = np.mat(np.identity(10754))
I_300 = np.mat(np.identity(419))
I_814 = np.mat(np.identity(8603))

F_t = F_0
Z_t = Z_0
Q = (1 - a) * I_300 + a * r

for i in range(1000):
    print(i)

    Z_all = np.full((10754, 419), 0.0)
    for m in range(len(Z_t)):
        Z_all[m] = Z_t[m]

    F_t1 = np.dot(((1 - B - B * u) * I_1164 + B * P), F_t) + B * u * Z_all
    # print(F_t1)

    F_train = F_t1[0:8603]
    Z_t1 = np.dot((F_train + g * np.dot(Y_train_random, np.transpose(Q))),
                  np.linalg.inv(I_300 + g * np.dot(Q, np.transpose(Q))))

    Z_t = Z_t1
    F_t = F_t1

    Y_final = np.dot(F_t[8603:], Q)
    Y_final = np.array(Y_final)
    f1_micro = f1_score(y_true=Y_u, y_pred=np.array((Y_final > 0), dtype=float), average='micro')
    f1_macro = f1_score(y_true=Y_u, y_pred=np.array((Y_final > 0), dtype=float), average='macro')
    f1_example = f1_score(y_true=Y_u, y_pred=np.array((Y_final > 0), dtype=float), average='samples')
    rankingloss = label_ranking_loss(y_true=Y_u, y_score=Y_final)
    writer.add_scalar('f1_score_micro_aminer', f1_micro, i)
    writer.add_scalar('f1_score_macro_aminer', f1_macro, i)
    writer.add_scalar('f1_example_aminer', f1_example, i)
    writer.add_scalar('ranking_loss_aminer', rankingloss, i)
