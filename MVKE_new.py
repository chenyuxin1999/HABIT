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
label_embedding = np.load('label_embedding_OAG.npy', allow_pickle=True)
label_embedding_normal = min_max_scaler.fit_transform(label_embedding)
# label_embedding_normal = normalize(label_embedding, axis=0)
label_embedding = torch.tensor(label_embedding_normal).to(torch.float32)
label_embedding = label_embedding.cuda()

writer = SummaryWriter('./MVKE_mag')

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
        self.E = 300
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
        #print(x)
        output_fc1 = self.fc1(x)  # K: (B,n,E)
        output_fc2 = self.fc2(self.vk)  # Q: (v,E)
        output_matmul1 = torch.matmul(output_fc1, output_fc2.transpose(1,0))/math.sqrt(self.dk)  # (B,n,v)
        #print(output_matmul1)

        output_sum = torch.sum(abs(x), dim=2)  # (B,n)
        output_sign = torch.sign(output_sum)  # (B,n)
        paddings = -2 ** 32 + 1
        paddings = torch.tensor(paddings).to(torch.float32).cuda()
        output_sign = torch.unsqueeze(output_sign, dim=2)  # (B,n,1)
        output_sign = output_sign.repeat(1,1,3)
        output = torch.where(output_sign == 0, paddings, output_matmul1)
        #print(output)
        output_softmax1 = self.softmax1(output)  # (B,n,v)
        output_weighted_sum1 = torch.matmul(output_softmax1.transpose(2,1), output_fc1)  # (B,v,E)
        output_fc3 = self.fc3(self.vk)  # K: (v,E)


        output_fc4 = self.fc4(tag_embedding)  # Q: (B,300,E)
        output_matmul2 = torch.matmul(output_fc4, output_fc3.transpose(1,0))/math.sqrt(self.dk)  # (B,300,v)
        output_softmax2 = self.softmax2(output_matmul2)  # (B,300,v)
        output_weighted_sum2 = torch.matmul(output_softmax2, output_weighted_sum1)  # (B,300,E)

        user_embedding = output_weighted_sum2  # (B,300,E)
        # print(user_embedding)
        inner_product = torch.mul(user_embedding, tag_embedding)  # (B,300,E)
        dot = torch.sum(inner_product, dim=2)  # (B,300)
        # print(dot)
        final_output = dot

        return final_output

def train():
    user_set = torch.load('keyword_embedding_merge.pt')
    user_label = torch.load('label_OAG.pt')

    user_index = torch.tensor(list(range(0, 10754)))

    # user_label = torch.ones(len(training_set), 1).cuda()

    data = GetLoader(user_index, user_label)
    train_data, eval_data = random_split(data, [round(0.8 * len(user_index)), round(0.2 * len(user_index))],
                                         generator=torch.Generator().manual_seed(42))

    dataTrain = DataLoader(dataset=train_data, batch_size=30, shuffle=True)
    dataTest = DataLoader(dataset=eval_data, batch_size=30, shuffle=True)
    # data_test = GetLoader(testData, testLabel)
    # dataTrain = DataLoader(data_train, batch_size=100, shuffle=True)
    # dataTest = DataLoader(data_test, batch_size=40, shuffle=False)

    sigmoid = nn.Sigmoid()

    model = MVKE().cuda()

    init_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    end_epoch = 500

    for epoch in range(end_epoch):
        all_loss_train = 0
        all_loss_test = 0
        f1_example = 0
        f1_macro = 0
        f1_micro = 0
        rankingloss = 0
        for i, (data, label) in enumerate(dataTrain):
            batch_keyword_embedding = []
            for j in data:
                batch_keyword_embedding.append(user_set[j])

            batch_keyword_embedding_padding = nn.utils.rnn.pad_sequence(batch_keyword_embedding, batch_first=True, padding_value=0)

            input, target = batch_keyword_embedding_padding.to(torch.float32).cuda(), label.to(torch.float32).cuda()
            model.train()

            optimizer.zero_grad()  # 使用之前先清零
            #print(input)

            output = model(input, label_embedding)
            # print(output.cpu().detach().ge(0.5).float().numpy())

            loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(output, target)

            loss.backward()  # loss反传，计算模型中各tensor的梯度
            optimizer.step()
            all_loss_train = all_loss_train + loss
        all_loss_train = all_loss_train/287
        # output = sigmoid(output)
        writer.add_scalar('MVKE_loss_train_merge', all_loss_train, epoch)

        with torch.no_grad():
            model.eval()
            for i, (data, label) in enumerate(dataTest):

                batch_keyword_embedding = []
                for j in data:
                    batch_keyword_embedding.append(user_set[j])

                batch_keyword_embedding_padding = nn.utils.rnn.pad_sequence(batch_keyword_embedding, batch_first=True,
                                                                            padding_value=0)

                input, target = batch_keyword_embedding_padding.to(torch.float32).cuda(), label.to(torch.float32).cuda()

                optimizer.zero_grad()  # 使用之前先清零

                output = model(input, label_embedding)
                # print(output.cpu().detach().ge(0.5).float().numpy())
                #print(output)

                loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(output, target)

                output = sigmoid(output)
                f1_micro_batch = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                                          y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='micro')
                f1_example_batch = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                                            y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='samples')
                f1_macro_batch = f1_score(y_true=target.cpu().clone().detach().ge(0.5).float().numpy(),
                                          y_pred=output.cpu().clone().detach().ge(0.5).float().numpy(), average='macro')
                rankingloss_batch = label_ranking_loss(y_true=target.cpu().detach().float().numpy(),
                                                 y_score=output.cpu().detach().float().numpy())
                f1_example = f1_example + f1_example_batch
                f1_macro = f1_macro + f1_macro_batch
                f1_micro = f1_micro + f1_micro_batch
                rankingloss = rankingloss + rankingloss_batch

                all_loss_test = all_loss_test + loss
        scheduler.step()
        all_loss_test = all_loss_test/72
        f1_micro = f1_micro / 72
        f1_macro = f1_macro / 72
        f1_example = f1_example / 72
        rankingloss = rankingloss / 72

        writer.add_scalar('f1_score_micro_MVKE_merge', f1_micro, epoch)
        writer.add_scalar('f1_score_macro_MVKE_merge', f1_macro, epoch)
        writer.add_scalar('f1_example_MVKE_merge', f1_example, epoch)
        writer.add_scalar('ranking_loss_MVKE_merge', rankingloss, epoch)
        writer.add_scalar('MVKE_loss_test_merge', all_loss_test, epoch)
        print("Epoch {} training loss:".format(epoch), all_loss_train)
        print("Epoch {} testing loss:".format(epoch), all_loss_test)
    print("\n*****  Training Done  *****")
    torch.save(model.state_dict(), 'MVKE_merge.pkl')


if __name__ == "__main__":
    train()






