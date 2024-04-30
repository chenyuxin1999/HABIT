import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.metrics import f1_score
from torch.utils.data import random_split
import math
from sklearn.metrics import label_ranking_loss

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
label_embedding = np.load('label_embedding.npy', allow_pickle=True)
label_embedding_normal = min_max_scaler.fit_transform(label_embedding)
# label_embedding_normal = normalize(label_embedding, axis=0)
label_embedding = np.array(label_embedding_normal.reshape(300, 100))
label_embedding = torch.tensor(label_embedding)
label_embedding = label_embedding.cuda()

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

class MVKE(nn.Module):
    def __init__(self):
        super(MVKE, self).__init__()
        self.E = 100
        self.V = 3
        self.C = 4
        self.dk = 100
        self.fc1 = nn.Linear(self.E, self.E)
        self.fc2 = nn.Linear(self.E, self.E)
        self.softmax1 = nn.Softmax(dim=1)
        self.fc3 = nn.Linear(self.E, self.E)
        self.fc4 = nn.Linear(self.E, self.E)
        self.softmax2 = nn.Softmax(dim=1)
        self.vk = nn.Parameter(torch.rand(self.V, self.E))

    def forward(self, x, tag_embedding):
        output_fc1 = self.fc1(x)  # K: (B,n,E)
        output_fc2 = self.fc2(self.vk)  # Q: (v,E)
        output_matmul1 = torch.matmul(output_fc1, output_fc2.transpose(1,0))/math.sqrt(self.dk)  # (B,n,v)

        output_sum = torch.sum(abs(x), dim=2)
        output_sign = torch.sign(output_sum)  # (B,n)
        paddings = -2 ** 32 + 1
        paddings = torch.tensor(paddings).to(torch.float32).cuda()
        output_sign = torch.unsqueeze(output_sign, dim=2)
        output_sign = output_sign.repeat(1,1,3)
        output = torch.where(output_sign == 0, paddings, output_matmul1)
        output_softmax1 = self.softmax1(output)  # (B,n,v)
        output_weighted_sum1 = torch.matmul(output_softmax1.transpose(2,1), output_fc1)  # (B,v,E)
        output_fc3 = self.fc3(self.vk)  # K: (v,E)


        output_fc4 = self.fc4(tag_embedding)  # Q: (B,300,E)
        output_matmul2 = torch.matmul(output_fc4, output_fc3.transpose(1,0))/math.sqrt(self.dk)  # (B,300,v)
        output_softmax2 = self.softmax2(output_matmul2)  # (B,300,v)
        output_weighted_sum2 = torch.matmul(output_softmax2, output_weighted_sum1)  # (B,300,E)

        user_embedding = output_weighted_sum2  # (B,300,E)
        inner_product = torch.mul(user_embedding, tag_embedding)  # (B,300,E)
        dot = torch.sum(inner_product, dim=2)  # (B,300)
        final_output = dot

        return final_output

def train():
    user_set = torch.load('input_feature_delicious.pt')
    user_set = user_set.cuda()
    user_label = torch.load('MVKE_user_label.pt')
    # user_label = torch.ones(len(training_set), 1).cuda()

    data = GetLoader(user_set, user_label)
    train_data, eval_data = random_split(data, [round(0.8 * user_set.shape[0]), round(0.2 * user_set.shape[0])],
                                         generator=torch.Generator().manual_seed(42))

    dataTrain = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
    dataTest = DataLoader(dataset=eval_data, batch_size=len(eval_data), shuffle=True)
    # data_test = GetLoader(testData, testLabel)
    # dataTrain = DataLoader(data_train, batch_size=100, shuffle=True)
    # dataTest = DataLoader(data_test, batch_size=40, shuffle=False)

    sigmoid = nn.Sigmoid()

    model = MVKE().cuda()
    model.load_state_dict(torch.load('MVKE_delicious_keyword.pkl'))
    model.eval()

    end_epoch = 1

    with torch.no_grad():
        for epoch in range(end_epoch):

            for i, (data, label) in enumerate(dataTest):

                input, target = data.to(torch.float32).cuda(), label.to(torch.float32).cuda()

                output = model(input, label_embedding)
                # print(output.cpu().detach().ge(0.5).float().numpy())
                #print(output)

                loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(output, target)

                all_loss_test = loss
            #all_loss_test = all_loss_test/2
            output = sigmoid(output)
            f1_micro = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                                y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='micro')
            f1_macro = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                                y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='macro')
            f1_example = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                                  y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='samples')
            rankingloss = label_ranking_loss(y_true=target.cpu().detach().float().numpy(), y_score=output.cpu().detach().float().numpy())
            print("Epoch {} testing loss:".format(epoch), all_loss_test)
    print("\n*****  Testing Done  *****")
    print(f1_example)
    print(f1_macro)
    print(f1_micro)
    print(rankingloss)


if __name__ == "__main__":
    train()
