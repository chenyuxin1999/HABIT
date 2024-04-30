import numpy as np
import csv
from datetime import datetime
import torch

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

with open("../data_new/user_label_num.csv", "r", encoding ="utf-8") as f:
    reader = csv.reader(f)
    user_label_num = [row for row in reader]
    del user_label_num[0]
    user_label_num = np.array(user_label_num)


Ids = np.array([val for val in userIds_de if val in userIds_tw])
user_label = []
for i in range(len(Ids)):
    label = np.zeros((300,1))
    label_num = user_label_num[user_label_num[:, 1] == Ids[i]]
    label_num = label_num[:, 2][0][1:-1].replace(" ","").split(',')
    for num in label_num:
        label[int(num)] = 1
    user_label.append(label)

user_label = np.array(user_label, dtype=object)
np.save('../userlabel/user_label_vectory.npy', user_label)


