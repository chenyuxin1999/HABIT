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
import math

label_embedding = np.load('label_embedding.npy', allow_pickle=True)
label_embedding_normal = normalize(label_embedding, axis=1)
np.array(label_embedding_normal.reshape(300, 100))
label_embedding = torch.tensor(label_embedding)
label_embedding = label_embedding.cuda()

MVKE_user_post = torch.load('MVKE_user_post.pt')
print(np.shape(MVKE_user_post))
all_user_post = {}
for i in range(len(MVKE_user_post)):
    all_user_post[i] = MVKE_user_post[i]
all_user_post ={key : all_user_post[key].cuda() for key in all_user_post}

writer = SummaryWriter('./tensorboard')

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
        self.n = 90
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
        output = torch.where(output_sign == 0, output_matmul1, paddings)
        #print('111')
        #for z in range(len(output_sign)):
        #    for j in range(len(output_sign[z])):
        #        if output_sign[z][j] == 0:
        #            output_matmul1[z][j] = -2 ** 32 + 1

        #print('222')
        #print(np.shape(output))
        output_softmax1 = self.softmax1(output)  # (B,n,v)
        output_weighted_sum1 = torch.matmul(output_softmax1.transpose(2,1), output_fc1)  # (B,v,E)
        output_fc3 = self.fc3(self.vk)  # K: (v,E)


        output_fc4 = self.fc4(tag_embedding)  # Q: (B,E)
        output_matmul2 = torch.matmul(output_fc4, output_fc3.transpose(1,0))/math.sqrt(self.dk)  # (B,v)
        output_softmax2 = self.softmax2(output_matmul2)  # (B,v)
        output_softmax2 = output_softmax2.reshape(output_softmax2.size()[0], output_softmax2.size()[1],1)  # (B,v,1)
        output_weighted_sum2 = torch.matmul(output_weighted_sum1.transpose(2,1), output_softmax2)  # (B,E,1)

        user_embedding = torch.squeeze(output_weighted_sum2)  # (B,E)
        mul = torch.mul(user_embedding, tag_embedding)
        dot = torch.sum(mul, dim=1)
        final_output = torch.unsqueeze(dot, dim=1)

        return final_output

def train():
    training_set = torch.load('training_set.pt')
    training_set = training_set.cuda()
    # user_label = torch.load('MVKE_user_label.pt')
    user_label = torch.ones(len(training_set), 1).cuda()
    data_train = GetLoader(training_set, user_label)
    # data_test = GetLoader(testData, testLabel)
    dataTrain = DataLoader(data_train, batch_size=512, shuffle=True)
    # dataTest = DataLoader(data_test, batch_size=40, shuffle=False)

    sigmoid = nn.Sigmoid()
    ## model
    model = MVKE().cuda()

    init_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    end_epoch = 3000

    for epoch in range(end_epoch):
        all_loss_train = 0
        all_loss_test = 0
        for i, (data, label) in enumerate(dataTrain):
            print(i)
            tag_embedding = []
            for j in range(len(data)):
                tag_embedding.append(label_embedding[int(data[j][1])].cpu().numpy())
            tag_embedding = torch.tensor(tag_embedding).cuda()

            user_post_embedding = []
            for k in range(len(data)):
                user_post_embedding.append(all_user_post[int(data[k][0])].cpu().numpy())
            user_post_embedding = torch.tensor(user_post_embedding).cuda()

            input, target = user_post_embedding.to(torch.float32), label.to(torch.float32)
            model.train()

            optimizer.zero_grad()  # 使用之前先清零

            output = model(input, tag_embedding)
            # print(output.cpu().detach().ge(0.5).float().numpy())

            loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(output, target)

            loss.backward()  # loss反传，计算模型中各tensor的梯度
            optimizer.step()
            all_loss_train = all_loss_train + loss
        all_loss_train = all_loss_train/145
        output = sigmoid(output)
        writer.add_scalar('loss--train', all_loss_train, epoch)
        print("Epoch {} training loss:".format(epoch), all_loss_train)
        print("Epoch {} testing loss:".format(epoch), all_loss_test)
    print("\n*****  Training Done  *****")
    torch.save(model.state_dict(), 'MVKE.pkl')


if __name__ == "__main__":
    train()






