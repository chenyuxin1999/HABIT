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

writer = SummaryWriter('./MVKE_aminer')

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
        self.n = 90
        self.dk = 300
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


        output_fc4 = self.fc4(tag_embedding)  # Q: (B,490k,E)
        output_matmul2 = torch.matmul(output_fc4, output_fc3.transpose(1,0))/math.sqrt(self.dk)  # (B,300,v)
        output_softmax2 = self.softmax2(output_matmul2)  # (B,300,v)
        output_weighted_sum2 = torch.matmul(output_softmax2, output_weighted_sum1)  # (B,300,E)

        user_embedding = output_weighted_sum2  # (B,300,E)
        # print(user_embedding)
        inner_product = torch.mul(user_embedding, tag_embedding)  # (B,300,E)
        final_output = torch.sum(inner_product, dim=2)  # (B,300)

        return final_output

def train():

    all_label_embedding = torch.load('all_label_embedding.pt')
    test_num_label_index = np.load('test_num_label_index.npy', allow_pickle=True).item()
    min_max_scaler = preprocessing.MinMaxScaler()
    all_label_embedding = min_max_scaler.fit_transform(all_label_embedding)
    all_label_embedding = torch.tensor(all_label_embedding).to(torch.float32).cuda()

    input_feature = torch.load('MVKE_input_aminer.pt')

    user_label = torch.tensor(list(range(0, 21789)))

    data = GetLoader(input_feature, user_label)
    train_data, eval_data = random_split(data,
                                         [round(0.8 * input_feature.shape[0]), round(0.2 * input_feature.shape[0])],
                                         generator=torch.Generator().manual_seed(42))

    dataTrain = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
    dataTest = DataLoader(dataset=eval_data, batch_size=100, shuffle=True)
    # data_test = GetLoader(testData, testLabel)
    # dataTrain = DataLoader(data_train, batch_size=100, shuffle=True)
    # dataTest = DataLoader(data_test, batch_size=40, shuffle=False)

    sigmoid = nn.Sigmoid()

    model = MVKE().cuda()

    init_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

    end_epoch = 100

    for epoch in range(end_epoch):
        all_loss_train = 0
        all_loss_test = 0
        f1_example = 0
        f1_macro = 0
        f1_micro = 0
        for i, (data, label) in enumerate(dataTrain):

            batch_label = torch.zeros(len(data), 488469)

            for j in range(len(data)):
                label_index = test_num_label_index[j]
                for k in label_index:
                    batch_label[j][k] = 1

            input, target = data.to(torch.float32).cuda(), batch_label.to(torch.float32).cuda()
            model.train()

            optimizer.zero_grad()  # 使用之前先清零
            #print(input)

            output = model(input, all_label_embedding)
            # print(output.cpu().detach().ge(0.5).float().numpy())

            loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(output, target)

            loss.backward()  # loss反传，计算模型中各tensor的梯度
            optimizer.step()
            all_loss_train = all_loss_train + loss
        all_loss_train = all_loss_train/175
        writer.add_scalar('MVKE_loss_train_aminer', all_loss_train, epoch)

        with torch.no_grad():
            for i, (data, label) in enumerate(dataTest):

                batch_label = torch.zeros(len(data), 488469)

                for j in range(len(data)):
                    label_index = test_num_label_index[j]
                    for k in label_index:
                        batch_label[j][k] = 1

                input, target = data.to(torch.float32).cuda(), batch_label.to(torch.float32).cuda()
                model.train()

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
                f1_example = f1_example + f1_example_batch
                f1_macro = f1_macro + f1_macro_batch
                f1_micro = f1_micro + f1_micro_batch

                all_loss_test = all_loss_test + loss
            all_loss_test = all_loss_test/44
            f1_micro = f1_micro/44
            f1_macro = f1_macro/44
            f1_example = f1_example/44

            writer.add_scalar('f1_score_micro_MVKE_aminer', f1_micro, epoch)
            writer.add_scalar('f1_score_macro_MVKE_aminer', f1_macro, epoch)
            writer.add_scalar('f1_example_MVKE_aminer', f1_example, epoch)
            writer.add_scalar('MVKE_loss_test_aminer', all_loss_test, epoch)
        print("Epoch {} training loss:".format(epoch), all_loss_train)
        print("Epoch {} testing loss:".format(epoch), all_loss_test)
    print("\n*****  Training Done  *****")
    torch.save(model.state_dict(), 'MVKE_aminer.pkl')


if __name__ == "__main__":
    train()
