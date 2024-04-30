import numpy as np
import csv
from datetime import datetime
import time
import torch

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(n_position[i]) for i in range(len(n_position))])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  偶数正弦
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  奇数余弦

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    # return torch.FloatTensor(sinusoid_table)  # n_position × d_hid  得到每一个词的位置向量
    return sinusoid_table


with open("../data_new/delicious_post.csv", "r", encoding ="utf-8") as f:
    reader = csv.reader(f)
    userinfos_de = [row for row in reader]
    del userinfos_de[0]
    #获取delicious网络下所有用户的关键词列表
    userinfos_de = np.array(userinfos_de)
    #获取delicious网络下所有用户id列表
    userIds_de = userinfos_de[:,0]

with open("../data_new/twitter_post.csv", "r", encoding ="utf-8") as f:
    reader = csv.reader(f)
    userinfos_tw = [row for row in reader]
    del userinfos_tw[0]
    #获取twitter网络下所有用户的关键词列表
    userinfos_tw = np.array(userinfos_tw)
    #获取twitter网络下所有用户id列表
    userIds_tw = userinfos_tw[:,0]

Ids = np.array([val for val in userIds_de if val in userIds_tw])


edge_size = 60
with open("../data_new/delicious用户post文件.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    postsinfos_de = [row for row in reader]
    del postsinfos_de[0]
    postsinfos_de = np.array(postsinfos_de)

    per_5_users_time_encode_de = []
    for i in range(len(Ids)):
        all_time = []
        id_time_arr = postsinfos_de[postsinfos_de[:, 1] == Ids[i]]
        id_time_arr = id_time_arr[:,2]
        if len(id_time_arr) > 100:
            id_time_arr = id_time_arr[0:100]
        for m in range(len(id_time_arr)):
            time_object = time.strptime(id_time_arr[m], '%Y-%m-%d %H:%M:%S')
            all_time.append(int(time.mktime(time_object)))
        delta = 0
        if len(all_time) == 1:
            delta = all_time[0]
            pos = np.array(all_time) / delta
        else:
            for n in range(0, len(all_time) - 1):
                delta += (all_time[n + 1] - all_time[n])

            delta = delta / (len(all_time) - 1)
            pos = (np.array(all_time) - np.array(all_time[0])) / delta
        # print(delta)
        # print(delta / (len(all_time)-1))
        # print((np.array(all_time) - np.array(all_time[0])) / delta)
        pos = np.floor(pos)
        pe = get_sinusoid_encoding_table(pos,edge_size)
        if len(pos) < 100:
            # zero = torch.zeros((100 - len(pos)), edge_size)
            # pe = torch.cat((pe,zero),0)
            zero = np.zeros(((100 - len(pos)), edge_size))
            pe = np.concatenate((pe, zero), axis=0)
        per_5_users_time_encode_de.append(pe)

    per_5_users_time_encode_de = np.array(per_5_users_time_encode_de)
    np.save('../time_encode/time_encode_de.npy', per_5_users_time_encode_de)


with open("../data_new/twitter用户post文件.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    postsinfos_tw = [row for row in reader]
    del postsinfos_tw[0]
    postsinfos_tw = np.array(postsinfos_tw)

    per_5_users_time_encode_tw = []
    for i in range(len(Ids)):
        all_time = []
        id_time_arr = postsinfos_tw[postsinfos_tw[:, 1] == Ids[i]]
        id_time_arr = id_time_arr[:,2]
        if len(id_time_arr) > 100:
            id_time_arr = id_time_arr[0:100]
        for m in range(len(id_time_arr)):
            time_object = time.strptime(id_time_arr[m], '%Y-%m-%d %H:%M:%S')
            all_time.append(int(time.mktime(time_object)))
        delta = 0
        if len(all_time) == 1:
            delta = all_time[0]
            pos = np.array(all_time) / delta
        else:
            for n in range(0, len(all_time) - 1):
                delta += (all_time[n + 1] - all_time[n])

            delta = delta / (len(all_time) - 1)
            pos = (np.array(all_time) - np.array(all_time[0])) / delta
        # print(delta)
        # print(delta / (len(all_time)-1))
        # print((np.array(all_time) - np.array(all_time[0])) / delta)
        pos = np.floor(pos)
        pe = get_sinusoid_encoding_table(pos,edge_size)
        if len(pos) < 100:
            zero = np.zeros(((100 - len(pos)), edge_size))
            pe = np.concatenate((pe, zero), axis=0)
        per_5_users_time_encode_tw.append(pe)

    per_5_users_time_encode_tw = np.array(per_5_users_time_encode_tw)
    np.save('../time_encode/time_encode_tw.npy', per_5_users_time_encode_tw)
