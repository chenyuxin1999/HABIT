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

min_max_scaler = preprocessing.MinMaxScaler()
label_embedding = np.load('label_embedding.npy', allow_pickle=True)
label_embedding_normal = min_max_scaler.fit_transform(label_embedding)
label_embedding = np.transpose(np.array(label_embedding_normal.reshape(300, 100)))
label_embedding = torch.tensor(label_embedding)
label_embedding = label_embedding.cuda()

writer = SummaryWriter('./tensorboard')


class CrossFeature(nn.Module):
    def __init__(self):
        super(CrossFeature, self).__init__()
        self.C = 4
        self.E = 100
        self.v = nn.Parameter(torch.rand(self.E, self.C))

    def forward(self, x):
        batch_c = []
        for i in range(x.shape[1] - 1):
            for j in range(i + 1, x.shape[1]):
                vi_vj = torch.dot(self.v[i], self.v[j])
                batch_xi_xj = x[:, i] * x[:, j]
                batch_c.append((batch_xi_xj * vi_vj).cpu().detach().numpy())
        batch_c = torch.tensor(batch_c).cuda()

        return batch_c.T


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


cross_feature = CrossFeature()


class UTPM(nn.Module):
    def __init__(self):
        super(UTPM, self).__init__()
        self.E = 100
        self.U = 100
        self.C = 4
        self.T = 8
        self.F = 4
        self.N = 50
        self.batchsize = 50
        self.fc1 = nn.Linear(self.E, self.T)
        self.fc2 = nn.Linear(self.T, 1, bias=False)
        self.softmax1 = nn.Softmax(dim=1)
        self.fc3 = nn.Linear(self.E, self.T)
        self.fc4 = nn.Linear(self.T, 1, bias=False)
        self.softmax2 = nn.Softmax(dim=1)
        self.fc5 = nn.Linear(2 * self.E, self.E)
        self.fc6 = nn.Linear(self.E, self.E)
        self.fc7 = nn.Linear(int(self.E * (self.E - 1) / 2), self.E)
        self.fc8 = nn.Linear(self.E * 2, self.U)
        self.relu = nn.ReLU()

    def forward(self, x):
        output_fc1 = self.fc1(x)
        output_fc2 = self.fc2(output_fc1)
        output_fc2 = torch.exp(output_fc2)
        output_softmax1 = self.softmax1(output_fc2)
        output_softmax1 = output_softmax1.reshape([output_softmax1.size()[0], output_softmax1.size()[1], 1])
        output_first_query = torch.matmul(x.transpose(1, 2), output_softmax1, out=None).squeeze()

        output_fc3 = self.fc3(x)
        output_fc4 = self.fc4(output_fc3)
        output_fc4 = torch.exp(output_fc4)
        output_softmax2 = self.softmax2(output_fc4)
        output_softmax2 = output_softmax2.reshape([output_softmax2.size()[0], output_softmax2.size()[1], 1])
        output_second_query = torch.matmul(x.transpose(1, 2), output_softmax2, out=None).squeeze()

        output_concatenate1 = torch.cat((output_first_query, output_second_query), dim=1)
        output_fc5 = self.fc5(output_concatenate1)
        output_fc6 = self.fc6(output_fc5)

        output_crossfeature = cross_feature(output_fc5)
        output_fc7 = self.fc7(output_crossfeature)

        output_concatenate2 = torch.cat((output_fc6, output_fc7), dim=1)
        output_fc8 = self.fc8(output_concatenate2)

        final_output = torch.mm(output_fc8, label_embedding)

        return final_output


def train():
    input_feature = torch.load('input_feature_delicious.pt')
    input_feature = torch.tensor(input_feature).to(torch.float32).cuda()
    user_label = torch.load('MVKE_user_label.pt').to(torch.float32).cuda()
    user_label = torch.tensor(user_label).to(torch.float32).cuda()

    print(np.shape(input_feature))
    print(np.shape(user_label))

    data = GetLoader(input_feature, user_label)
    train_data, eval_data = random_split(data, [round(0.8 * input_feature.shape[0]), round(0.2 * input_feature.shape[0])],
                                         generator=torch.Generator().manual_seed(42))

    dataTrain = DataLoader(dataset=train_data, batch_size=40, shuffle=True)
    dataTest = DataLoader(dataset=eval_data, batch_size=40, shuffle=True)


    sigmoid = nn.Sigmoid()
    ## model
    model = UTPM().cuda()

    init_lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

    end_epoch = 3000
    ### start train
    for epoch in range(end_epoch):
        all_loss_train = 0
        all_loss_test = 0
        for i, (data, label) in enumerate(dataTrain):
            input, target = data, label
            model.train()

            optimizer.zero_grad()  # 使用之前先清零

            output = model(input)
            #print(output)
            # print(output.cpu().detach().ge(0.5).float().numpy())
            output = torch.mm(output, label_embedding)

            loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(output, target)

            loss.backward()  # loss反传，计算模型中各tensor的梯度
            optimizer.step()
            all_loss_train = all_loss_train + loss
        all_loss_train = all_loss_train / 24
        output = sigmoid(output)
        f1_micro = f1_score(y_true=target.cpu().detach().clone().ge(0.5).float().numpy(),
                            y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='micro')
        f1_macro = f1_score(y_true=target.cpu().detach().clone().ge(0.5).float().numpy(),
                            y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='macro')
        f1_example = f1_score(y_true=target.cpu().detach().clone().ge(0.5).float().numpy(),
                              y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='samples')
        #writer.add_scalar('f1_score_micro_train_UTPM_delicious', f1_micro, epoch)
        #writer.add_scalar('f1_score_macro_train_UTPM_delicious', f1_macro, epoch)
        #writer.add_scalar('f1_example_train_UTPM_delicious', f1_example, epoch)

        for i, (data, label) in enumerate(dataTest):
            input, target = data, label
            optimizer.zero_grad()  # 使用之前先清零

            output = model(input)
            #print(output)
            output = torch.mm(output, label_embedding)

            loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(output, target)

            all_loss_test = all_loss_test + loss
        all_loss_test = all_loss_test / 6
        print(output)
        output = sigmoid(output)
        f1_micro = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                            y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='micro')
        f1_macro = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                            y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='macro')
        f1_example = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                              y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='samples')
        #writer.add_scalar('f1_score_micro_test_UTPM_delicious', f1_micro, epoch)
        #writer.add_scalar('f1_score_macro_test_UTPM_delicious', f1_macro, epoch)
        #writer.add_scalar('f1_example_test_UTPM_delicious', f1_example, epoch)

        #writer.add_scalar('loss_train_UTPM_delicious', all_loss_train, epoch)
        #writer.add_scalar('loss_test_UTPM_delicious', all_loss_test, epoch)
        print("Epoch {} training loss:".format(epoch), all_loss_train)
        print("Epoch {} testing loss:".format(epoch), all_loss_test)
    print("\n*****  Training Done  *****")
    torch.save(model.state_dict(), 'UTPM_delicious.pkl')


if __name__ == "__main__":
    train()
