import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

writer = SummaryWriter('./LSTM')


h_values = {}
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshaped)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.contiguous().view(-1, x.size(1), y.size(-1))
        return y

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


class FishNet_MaskOutput(nn.Module):
    def __init__(self, lengthVec):
        super(FishNet_MaskOutput, self).__init__()
        self.lstm1 = nn.LSTM(300, 256, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(256, 300, batch_first=True, bidirectional=False)
        self.encoder_fc = nn.Linear(256,256)
        self.decoder_fc = nn.Linear(256,256)
        self.relu = nn.ReLU()
        self.lengthVec = lengthVec
        self.maxLength = 500

    def forward(self, x, i, h_value):
        #print(np.shape(x))
        #print(i)
        #print(np.shape(self.lengthVec[i*50:(i+1)*50]))
        # Input X with padding
        # LSTM1 with masking "0"
        block1 = nn.utils.rnn.pack_padded_sequence(input=x, lengths=self.lengthVec[i*200:(i+1)*200],
                                                         batch_first=True, enforce_sorted=False)
        block1, _ = self.lstm1(block1)
        out, out1 = nn.utils.rnn.pad_packed_sequence(block1, batch_first=True,
                                                           total_length=self.maxLength)

        for k, v in zip(out, self.lengthVec[i*200:(i+1)*200]):
            h_value.append(k[v - 1].cpu().detach().numpy())

        #out = self.encoder_fc(out)
        #out = self.relu(out)
        #out = self.decoder_fc(out)
        #out = self.relu(out)

        #LSTM2 & TimeDistributed
        out, _ = self.lstm2(out)

        # Use self.lengthVec construct output mask
        #for i in range(len(self.lengthVec)):
        #    out2[i, self.lengthVec[i]:, :] = 0
        #print(np.shape(out2))

        return out,h_value


lengthVec = np.load('length_mag.npy')

def train():
    # Create toy data
    # inputShape = np.shape(x_normal)
    #inputShape = [2,len(test[0]),100]
    inputData = torch.load('author_post_padding_mag.pt')

    data_train = GetLoader(inputData,inputData)
    datas = DataLoader(data_train, batch_size=200, shuffle=False)

    ## model
    model = FishNet_MaskOutput(lengthVec).cuda()

    init_lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.90)
    #input = torch.tensor(inputData).to(torch.float32).cuda()



    end_epoch = 501
    for epoch in range(end_epoch):
        all_loss = 0
        h_value = []
        for i, (data,label) in enumerate(datas):
            input, target = torch.tensor(data).to(torch.float32).cuda(), torch.tensor(data).to(torch.float32).cuda()
            model.train()

            optimizer.zero_grad()

            output, h_value = model(input,i,h_value)

            output1 = output.clone()

            for j in range(len(lengthVec[i*200:(i+1)*200])):
                output1[j, lengthVec[i*200+j]:, :] = 0

            loss = torch.nn.MSELoss(reduction = 'mean')(output1, input)

            loss.backward()  # loss反传，计算模型中各tensor的梯度
            optimizer.step()
            all_loss = all_loss + loss
        scheduler.step()
        if epoch == 200:
                np.save('h_values_mag_200.npy', h_value)
        if epoch == 300:
                np.save('h_values_mag_300.npy', h_value)
        if epoch == 400:
                np.save('h_values_mag_400.npy', h_value)
        if epoch == 500:
                np.save('h_values_mag_500.npy', h_value)
        all_loss = all_loss/54

        writer.add_scalar('loss_mag',all_loss , epoch)
        print("Epoch {} loss:".format(epoch), all_loss)
    print("\n*****  Training Done  *****")

if __name__ == "__main__":
    train()
