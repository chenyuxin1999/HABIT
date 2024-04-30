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
from sklearn.metrics import label_ranking_loss

writer = SummaryWriter('./UTPM_mag')


class CrossFeature(nn.Module):
    def __init__(self):
        super(CrossFeature, self).__init__()
        self.C = 4
        self.E = 300
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
        self.E = 300
        self.U = 300
        self.C = 4
        self.T = 8
        self.F = 4
        self.N = 50
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
        self.dropout = nn.Dropout(p=0.1)
        self.BatchNorm = nn.BatchNorm1d(self.E)
        self.BatchNorm2 = nn.BatchNorm1d(self.E)
        self.BatchNorm3 = nn.BatchNorm1d(self.E)
        self.q1 = nn.Parameter(torch.rand(self.T))
        self.q2 = nn.Parameter(torch.rand(self.T))



    def forward(self, x, label_embedding):
        BatchNorm1 = nn.BatchNorm1d(x.size()[1]).cuda()
        BatchNorm4 = nn.BatchNorm1d(x.size()[1]).cuda()

        output_fc1 = self.fc1(x)
        output_fc1 = self.relu(output_fc1)
        output_fc1 = BatchNorm1(output_fc1)

        output_fc1 = torch.matmul(output_fc1, self.q1)
        #output_fc1 = self.dropout(output_fc1)

        # output_fc2 = self.fc2(output_fc1)
        # output_fc2 = BatchNorm1(output_fc2)
        #output_fc2 = self.dropout(output_fc2)
        # output_fc2 = torch.exp(output_fc2)

        output_sum = torch.sum(abs(x), dim=2)
        output_sign = torch.sign(output_sum)  # (B,n)
        paddings = -2 ** 32 + 1
        paddings = torch.tensor(paddings).to(torch.float32).cuda()
        # output_sign = torch.unsqueeze(output_sign, dim=2)
        output_mask1 = torch.where(output_sign == 0, paddings, output_fc1)

        output_softmax1 = self.softmax1(output_mask1)
        output_softmax1 = output_softmax1.reshape([output_softmax1.size()[0], output_softmax1.size()[1], 1])
        output_first_query = torch.matmul(x.transpose(1, 2), output_softmax1, out=None).squeeze()

        output_fc3 = self.fc3(x)
        output_fc3 = self.relu(output_fc3)
        output_fc3 = BatchNorm4(output_fc3)

        output_fc3 = torch.matmul(output_fc3, self.q2)
        #output_fc3 = self.dropout(output_fc3)


        # output_fc4 = self.fc4(output_fc3)
        # output_fc4 = BatchNorm1(output_fc4)
        #output_fc4 = self.dropout(output_fc4)
        # output_fc4 = torch.exp(output_fc4)

        output_mask2 = torch.where(output_sign == 0, paddings, output_fc3)

        output_softmax2 = self.softmax2(output_mask2)
        output_softmax2 = output_softmax2.reshape([output_softmax2.size()[0], output_softmax2.size()[1], 1])
        output_second_query = torch.matmul(x.transpose(1, 2), output_softmax2, out=None).squeeze()

        output_concatenate1 = torch.cat((output_first_query, output_second_query), dim=1)
        output_fc5 = self.fc5(output_concatenate1)
        output_fc5 = self.BatchNorm(output_fc5)
        #output_fc5 = self.dropout(output_fc5)

        output_fc6 = self.fc6(output_fc5)
        output_fc6 = self.BatchNorm2(output_fc6)
        #output_fc6 = self.dropout(output_fc6)

        output_crossfeature = cross_feature(output_fc5)
        output_fc7 = self.fc7(output_crossfeature)
        output_fc7 = self.BatchNorm3(output_fc7)
        #output_fc7 = self.dropout(output_fc7)

        output_concatenate2 = torch.cat((output_fc6, output_fc7), dim=1)
        output_fc8 = self.fc8(output_concatenate2)
        print(output_fc8)

        final_output = torch.matmul(output_fc8, label_embedding.transpose(-1, -2))

        return final_output


def train():
    # label_embedding = np.load('user_tags_embeddings.npy', allow_pickle=True)
    # label_index = np.load('user_tags.npy', allow_pickle=True)
    # training_label = np.load('random_label.npy', allow_pickle=True)
    # index_label_embedding = np.load('index_label_embedding.npy', allow_pickle=True)
    keyword_embedding = torch.load('keyword_embedding_mag.pt')
    # user_label_matrix = torch.load('user_label_matrix.pt')
    # normalized_label_embedding = torch.load('random_label_embedding.pt')
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    label_embedding = np.load('label_embedding_OAG.npy', allow_pickle=True)
    label_embedding_normal = min_max_scaler.fit_transform(label_embedding)
    label_embedding = torch.tensor(label_embedding_normal).to(torch.float32)
    label_embedding = label_embedding.cuda()

    input_feature = torch.tensor(list(range(0, 10754)))
    user_label = torch.load('label_OAG.pt')


    data = GetLoader(input_feature, user_label)
    train_data, eval_data = random_split(data,
                                         [round(0.8 * input_feature.shape[0]), round(0.2 * input_feature.shape[0])],
                                         generator=torch.Generator().manual_seed(42))

    dataTrain = DataLoader(dataset=train_data, batch_size=500, shuffle=True)
    dataTest = DataLoader(dataset=eval_data, batch_size=500, shuffle=True)

    sigmoid = nn.Sigmoid()
    # model
    model = UTPM().cuda()

    init_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.90)

    end_epoch = 300
    # start train
    for epoch in range(end_epoch):
        all_loss_train = 0
        all_loss_test = 0
        f1_micro = 0
        f1_macro = 0
        f1_example = 0
        rankingloss = 0

        model.train()

        for i, (data, label) in enumerate(dataTrain):
            batch_keyword_embedding = []
            for j in data:
                batch_keyword_embedding.append(keyword_embedding[j])

            batch_keyword_embedding_padding = nn.utils.rnn.pad_sequence(batch_keyword_embedding, batch_first=True,
                                                                        padding_value=0)

            input, target = batch_keyword_embedding_padding.to(torch.float32).cuda(), label.to(torch.float32).cuda()
            model.train()

            optimizer.zero_grad()  # 使用之前先清零
            # print(input)

            output = model(input, label_embedding)
            # print(output.cpu().detach().ge(0.5).float().numpy())

            loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(output, target)

            loss.backward()  # loss反传，计算模型中各tensor的梯度

            optimizer.step()
            output = sigmoid(output)
            print(output.cpu().clone().detach().ge(0.5).float())
            print(torch.sum(output.cpu().clone().detach().ge(0.5).float()))
            all_loss_train = all_loss_train + loss
        all_loss_train = all_loss_train / 18
        scheduler.step()

        #if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            model.eval()
            for i, (data, label) in enumerate(dataTest):

                batch_keyword_embedding = []
                for j in data:
                    batch_keyword_embedding.append(keyword_embedding[j])

                batch_keyword_embedding_padding = nn.utils.rnn.pad_sequence(batch_keyword_embedding, batch_first=True,
                                                                            padding_value=0)

                input, target = batch_keyword_embedding_padding.to(torch.float32).cuda(), label.to(torch.float32).cuda()

                optimizer.zero_grad()  # 使用之前先清零

                output = model(input, label_embedding)
                # print(output.cpu().detach().ge(0.5).float().numpy())
                # print(output)

                loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(output, target)

                output = sigmoid(output)
                print(output.cpu().clone().detach().ge(0.5).float())
                print(torch.sum(output.cpu().clone().detach().ge(0.5).float()))
                f1_micro_batch = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                                          y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='micro')
                f1_example_batch = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                                            y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(),average='samples')
                f1_macro_batch = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                                          y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='macro')
                rankingloss_batch = label_ranking_loss(y_true=target.cpu().detach().float().numpy(),
                                                       y_score=output.cpu().detach().float().numpy())
                f1_example = f1_example + f1_example_batch
                f1_macro = f1_macro + f1_macro_batch
                f1_micro = f1_micro + f1_micro_batch
                rankingloss = rankingloss + rankingloss_batch

                all_loss_test = all_loss_test + loss
        all_loss_test = all_loss_test / 5
        f1_micro = f1_micro / 5
        f1_macro = f1_macro / 5
        f1_example = f1_example / 5
        rankingloss = rankingloss / 5
        writer.add_scalar('f1_score_micro_UTPM_mag', f1_micro, epoch)
        writer.add_scalar('f1_score_macro_UTPM_mag', f1_macro, epoch)
        writer.add_scalar('f1_example_UTPM_mag', f1_example, epoch)
        writer.add_scalar('loss_test_UTPM_mag', all_loss_test, epoch)
        writer.add_scalar('ranking_loss_UTPM_mag', rankingloss, epoch)
        print("Epoch {} testing loss:".format(epoch), all_loss_test)

        writer.add_scalar('loss_train_UTPM_mag', all_loss_train, epoch)
        print("Epoch {} training loss:".format(epoch), all_loss_train)
    print("\n*****  Training Done  *****")
    torch.save(model.state_dict(), 'UTPM_mag.pkl')


if __name__ == "__main__":
    train()
