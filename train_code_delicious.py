import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

a=np.load('wordHasVector_delicious.npy',allow_pickle=True)
wordHasVector=a.tolist()


delicious = pd.read_csv('delicious用户post文件.csv',encoding='utf8')
delicious.drop(axis=1, columns='Unnamed: 0',inplace=True)
twitter = pd.read_csv('twitter_group_by_userid.csv',encoding='utf8')
twitter.drop(axis=0, labels=1167,inplace=True)
twitter.drop(axis=0, labels=355,inplace=True)
twitter.drop(axis=0, labels=510,inplace=True)
twitter.drop(axis=0, labels=912,inplace=True)
twitter = twitter.reset_index()
userid_x = list(twitter['userid'])
for i in range(len(delicious['userid'])):
    if delicious['userid'][i] not in userid_x:
        delicious.drop(axis=0, labels=i,inplace=True)
delicious = delicious.reset_index()

user_post = []
all_user = []
embedding_sum = 0
for i in range(len(delicious['userid'])-1):
    for j in delicious['0'][i].replace("'","").replace("[","").replace(']','').split(','):
        try:
            embedding = wordHasVector[j.replace(' ','')]
        except:
            print(i)
            print(j)
        embedding_sum = embedding_sum + embedding
    post_embedding = embedding_sum/len(delicious['0'][i])
    user_post.append(post_embedding)
    if delicious['userid'][i+1] != delicious['userid'][i] or i == 93547:
        all_user.append(user_post)
        user_post = []


train_x = []
for i in all_user:
    tensor = torch.tensor(i)
    train_x.append(tensor)

x = nn.utils.rnn.pad_sequence(train_x, batch_first=True, padding_value=0)

writer = SummaryWriter('./tensorboard')


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
        self.lstm1 = nn.LSTM(100, 256, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(256, 100, batch_first=True, bidirectional=False)
        self.encoder_fc = nn.Linear(256,256)
        self.decoder_fc = nn.Linear(256,256)
        self.relu = nn.ReLU()
        self.lengthVec = lengthVec
        self.maxLength = 473

    def forward(self, x,i,h_value):
        #print(np.shape(x))
        #print(i)
        #print(np.shape(self.lengthVec[i*50:(i+1)*50]))
        # Input X with padding
        # LSTM1 with masking "0"
        block1 = nn.utils.rnn.pack_padded_sequence(input=x, lengths=self.lengthVec[i*32:(i+1)*32],
                                                         batch_first=True, enforce_sorted=False)
        block1, _ = self.lstm1(block1)
        out, out1 = nn.utils.rnn.pad_packed_sequence(block1, batch_first=True,
                                                           total_length=self.maxLength)

        for k, v in zip(out, self.lengthVec[i*32:(i+1)*32]):
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

        return out, h_value


lengthVec = []
for i in train_x:
    lengthVec.append(len(i))

x_reshape = x.numpy().reshape((-1,100))

x_normal = normalize(x_reshape,axis=1)
x_normal = x_normal.reshape((1164,473,100))

print(torch.cuda.is_available())
print(torch.cuda.device_count())

def train():
    # Create toy data
    inputShape = np.shape(x_normal)
    #inputShape = [2,len(test[0]),100]
    inputData = x_normal

    data_train = GetLoader(inputData,inputData)
    datas = DataLoader(data_train, batch_size=32, shuffle=False)

    ## model
    model = FishNet_MaskOutput(lengthVec).cuda()

    init_lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    #input = torch.tensor(inputData).to(torch.float32).cuda()



    end_epoch = 4001
    ### start train
    for epoch in range(end_epoch):
        all_loss = 0
        h_value = []
        for i, (data,label) in enumerate(datas):
            #print(np.shape(data))
            #print(i)
            train_loss = []
            input, target = torch.tensor(data).to(torch.float32).cuda(), torch.tensor(data).to(torch.float32).cuda()
            #print(np.shape(input))
            model.train()

            optimizer.zero_grad()  # 使用之前先清零

            output,h_value = model(input, i, h_value)

            output1 = output.clone()

            for j in range(len(lengthVec[i*32:(i+1)*32])):
                output1[j, lengthVec[i*32+j]:, :] = 0

            #print(output)
            loss = torch.nn.MSELoss(reduction='mean')(output1, input)

            loss.backward()  # loss反传，计算模型中各tensor的梯度
            optimizer.step()
            all_loss = all_loss + loss
        #print(np.linalg.norm(output[766][0].detach().numpy() - target[766][0].detach().numpy()))
        #print(np.linalg.norm(output[0] - target[0]))
        #norm = []
        #for i in range(len(output[0])):
        #    a = np.linalg.norm(output[766][i].detach().numpy() - target[766][i].detach().numpy())
        #    norm.append(a)
        #print(np.sum(norm))
        if epoch == 4000:
                #h_value = h_value.cpu()
                #h_value = np.array(h_value)
                #h_value = torch.tensor(h_value).cpu()
                np.save('h_values_delicious2.npy', h_value)
                #h_values[epoch] = h_value
        #print(np.shape(h_value))
        all_loss = all_loss/37

        writer.add_scalar('loss',all_loss , epoch)
        print("Epoch {} loss:".format(epoch), all_loss)
        #np.save('h_values.npy', h_values)
    print("\n*****  Training Done  *****")
    #np.save('h_values3.npy', h_values)
    print(output1)

if __name__ == "__main__":
    train()
