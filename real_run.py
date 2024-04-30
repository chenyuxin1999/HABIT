import argparse
import pickle
import time
import numpy as np
from model import *
from sklearn.utils import class_weight
import random
import warnings
import os
from torch.utils.data import DataLoader,random_split
from Mydata import *

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--inputsize', type=int, default=100, help='inputsize')
parser.add_argument('--outputsize', type=int, default=60, help='outputsize')
parser.add_argument('--nhead', type=int, default=6, help='number of heads')
parser.add_argument('--num_encoder_layers', type=int, default=6, help='number of encoder_layers')
parser.add_argument('--dim_feedforward', type=int, default=500, help='dim_feedforward')
parser.add_argument('--epoch', type=int, default=1000, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-6, help='l2 penalty')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--rand', type=int, default=1234, help='rand_seed')
parser.add_argument('--normalization', action='store_true', help='add a normalization layer to the end')
parser.add_argument('--use_LDA', action='store_true', help='use LDA to construct semantic hyperedge')


args = parser.parse_args()

SEED = args.rand
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



def main():
    label_embedding = np.load('../data_new/label_embedding.npy', allow_pickle=True)
    label_embedding = label_embedding.tolist()
    label_embedding = np.array(label_embedding)
    label_embedding = torch.Tensor(label_embedding)


    dataset = Mydata()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataLoader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, collate_fn=my_to_tensor)
    test_dataLoader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True, collate_fn=my_to_tensor)
    model = trans_to_cuda(HGNN_ATT(args, args.inputsize, args.outputsize, args.nhead, args.num_encoder_layers, args.dim_feedforward))
    writer = SummaryWriter("logs")
    for i in range(args.epoch):
        print("-----第{}轮训练开始-----".format(i + 1))
        train_loss = train_model(model, train_dataLoader, args, label_embedding)
        test_loss, test_pred, test_labels = test_model(model, test_dataLoader, args, label_embedding)
        print('Train Loss:%.4f' % (train_loss))
        print('Test Loss:%.4f' % (test_loss))
        writer.add_scalar("train_loss", train_loss.item(), i)
        writer.add_scalar("test_loss", test_loss.item(), i)

        if i % 5 == 0:
            macro_f1 = f1_score(test_labels, test_pred, average='macro')
            micro_f1 = f1_score(test_labels, test_pred, average='micro')
            sample_f1 = f1_score(test_labels, test_pred, average='samples')
            ranking_loss = label_ranking_loss(test_labels, test_pred)
            print("Macro F1-Score:")
            print(macro_f1)
            print("Micro F1-Score:")
            print(micro_f1)
            print("Example F1-Score:")
            print(sample_f1)
            print("Ranking loss:")
            print(ranking_loss)
            writer.add_scalar("Macro F1-Score", macro_f1.item(), i/5)
            writer.add_scalar("Micro F1-Score", micro_f1.item(), i/5)
            writer.add_scalar("Example F1-Score", sample_f1.item(), i/5)
            writer.add_scalar("Ranking loss", ranking_loss, i/5)

if __name__ == '__main__':
    main()