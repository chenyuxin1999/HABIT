import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Mydata(Dataset):
    def __init__(self):
        HG_de = np.load("../userHyperGraph_new/user_hyper_graph_de.npy", allow_pickle=True)
        word_de = np.load("../userKeyword_Embedding/uke_de.npy", allow_pickle=True)
        time_en_de = np.load("../time_encode/time_encode_de.npy", allow_pickle=True)
        HG_tw = np.load("../userHyperGraph_new/user_hyper_graph_tw.npy", allow_pickle=True)
        word_tw = np.load("../userKeyword_Embedding/uke_tw.npy", allow_pickle=True)
        time_en_tw = np.load("../time_encode/time_encode_tw.npy", allow_pickle=True)
        user_label = np.load("../userlabel/user_label_vectory.npy", allow_pickle=True)
        self.x = list(zip(HG_de, word_de, time_en_de, HG_tw, word_tw, time_en_tw, user_label))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)

def my_to_tensor(dataset):
    HG_de = []
    word_de = []
    time_en_de = []
    HG_tw = []
    word_tw = []
    time_en_tw = []
    user_label = []
    for data in dataset:
        HG_de.append(data[0])
        word_de.append(data[1])
        time_en_de.append(data[2])
        HG_tw.append(data[3])
        word_tw.append(data[4])
        time_en_tw.append(data[5])
        user_label.append(data[6])
    HG_de = torch.Tensor(HG_de)
    word_de = torch.Tensor(word_de)
    time_en_de = torch.Tensor(time_en_de)
    HG_tw = torch.Tensor(HG_tw)
    word_tw = torch.Tensor(word_tw)
    time_en_tw = torch.Tensor(time_en_tw)
    user_label = torch.Tensor(user_label)
    data1 = [HG_de, word_de, time_en_de, HG_tw, word_tw, time_en_tw, user_label]
    return data1


# dataset = Mydata()
# dataloader = DataLoader(dataset, batch_size = 5, shuffle=True,collate_fn=my_to_tensor)
# for d in dataloader:
#     x_de, H_de, time_encoder_de, x_tw, H_tw, time_encoder_tw, label_embedding = d
#     print(x_de.shape)
#     print(H_de.shape)
#     break