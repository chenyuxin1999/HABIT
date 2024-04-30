import math
import time

import numpy as np
import torch
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from newHGAT import *
from mytransformer import *
from Discriminator import *
import torchvision
from torch.utils.tensorboard import SummaryWriter
from Mydata import *
from sklearn.metrics import f1_score,label_ranking_loss


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HGNN_ATT(nn.Module):
    def __init__(self, opt, input_size, output_size, nhead, num_encoder_layers, dim_feedforward, dropout=0.3, B=4):
        super(HGNN_ATT, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.B = B
        self.nhead1 = nhead
        self.num_encoder_layers1 = num_encoder_layers
        self.dim_feedforward1 = dim_feedforward
        self.nhead2 = nhead
        self.num_encoder_layers2 = num_encoder_layers
        self.dim_feedforward2 = dim_feedforward
        self.gat1 = HyperGraphAttentionLayerSparse(output_size, output_size, alpha=0.2)
        self.gat2 = HyperGraphAttentionLayerSparse(output_size, output_size, alpha=0.2)
        self.gat_e = HyperGraphAttentionOutLayer(output_size, output_size, alpha=0.2)
        self.my_transformer_de_1 = MyTransformer(d_model=output_size, nhead=nhead,num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward)
        self.my_transformer_de_2 = MyTransformer(d_model=output_size, nhead=nhead,num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward)
        self.my_transformer_tw_1 = MyTransformer(d_model=output_size, nhead=nhead,num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward)
        self.my_transformer_tw_2 = MyTransformer(d_model=output_size, nhead=nhead,num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward)
        self.discriminator = Discriminator(input_size=input_size, output_size=output_size, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward)
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)


    def forward(self, x_de, H_de, time_encoder_de, x_tw, H_tw, time_encoder_tw, label_embedding):
        l_de = trans_to_cuda(nn.Linear(self.input_size, self.output_size))
        x_de = l_de(x_de)
        l_tw = trans_to_cuda(nn.Linear(self.input_size, self.output_size))
        x_tw = l_tw(x_tw)

        x_de = self.gat1(x_de, H_de)
        x_de = self.gat2(x_de, H_de)
        e_de = self.gat_e(x_de,H_de)

        x_tw = self.gat1(x_tw, H_tw)
        x_tw = self.gat2(x_tw, H_tw)
        e_tw = self.gat_e(x_tw, H_tw)

        index = torch.sum(H_de, dim=-1)
        zero_de = torch.where(index > 0, torch.zeros_like(index), torch.ones_like(index))
        zero_de = trans_to_cuda(zero_de).bool()
        zero_de_2 = zero_de.clone()
        zero_de_2 = trans_to_cuda(torch.cat([zero_de_2, trans_to_cuda(torch.zeros(H_de.shape[0], self.B))], dim=1)).bool()
        index = torch.sum(H_tw, dim=-1)
        zero_tw = torch.where(index > 0, torch.zeros_like(index), torch.ones_like(index))
        zero_tw = trans_to_cuda(zero_tw).bool()
        zero_tw_2 = zero_tw.clone()
        zero_tw_2 = trans_to_cuda(torch.cat([zero_tw_2, trans_to_cuda(torch.zeros(H_tw.shape[0], self.B))], dim=1)).bool()


        # zero_de = trans_to_cuda(torch.zeros((H_de.shape[0], H_de.shape[1]))).bool()
        # zero_de_2 = trans_to_cuda(torch.zeros((H_de.shape[0], H_de.shape[1] + self.B))).bool()
        #
        # zero_tw = trans_to_cuda(torch.zeros((H_tw.shape[0], H_tw.shape[1]))).bool()
        # zero_tw_2 = trans_to_cuda(torch.zeros((H_tw.shape[0], H_tw.shape[1] + self.B))).bool()
        #
        # for i in range(H_de.shape[0]):
        #     for j in range(H_de.shape[1]):
        #         index = H_de[i][j].nonzero()
        #         if len(index) == 0:
        #             zero_de[i][j] = 1
        #             zero_de_2[i][j] = 1
        # for i in range(H_tw.shape[0]):
        #     for j in range(H_tw.shape[1]):
        #         index = H_tw[i][j].nonzero()
        #         if len(index) == 0:
        #             zero_tw[i][j] = 1
        #             zero_tw_2[i][j] = 1

        e_4tra_de = e_de + time_encoder_de
        e_4tra_de = e_4tra_de.transpose(0, 1)
        out = self.my_transformer_de_1(src=e_4tra_de, src_key_padding_mask=zero_de)
        out = self.my_transformer_de_1(src=out, src_key_padding_mask=zero_de)
        z_de = e_4tra_de + out

        e_4tra_tw = e_tw + time_encoder_tw
        e_4tra_tw = e_4tra_tw.transpose(0, 1)
        out = self.my_transformer_tw_1(src=e_4tra_tw, src_key_padding_mask=zero_tw)
        out = self.my_transformer_tw_1(src=out, src_key_padding_mask=zero_tw)
        z_tw = e_4tra_tw + out

        z_fns = trans_to_cuda(torch.randn((self.B, H_de.shape[0], self.output_size)))
        input_de = torch.cat((z_de, z_fns), 0)

        out2_de = self.my_transformer_de_2(src=input_de, src_key_padding_mask=zero_de_2)
        z_fns = out2_de[-4:, :, :]

        input_tw = torch.cat((z_tw, z_fns), 0)
        out2_tw = self.my_transformer_tw_2(src=input_tw, src_key_padding_mask=zero_tw_2)
        z_fns = out2_tw[-4:, :, :]

        input_de = torch.cat((out2_de[0:-4, :, :], z_fns), 0)
        out3_de = self.my_transformer_de_2(src=input_de, src_key_padding_mask=zero_de_2)
        z_fns = out3_de[-4:, :, :]

        de_out = out3_de[0:-4, :, :]
        input_tw = torch.cat((out2_tw[0:-4, :, :], z_fns), 0)
        out3_tw = self.my_transformer_tw_2(src=input_tw, src_key_padding_mask=zero_tw_2)

        tw_out = out3_tw[0:-4, :, :]
        out = self.discriminator(de_out=de_out, tw_out=tw_out, F0=label_embedding)
        return out


def forward(model, user_label, x_de, H_de, time_encoder_de, x_tw, H_tw, time_encoder_tw, label_embedding):
    x_de = trans_to_cuda(torch.Tensor(x_de))
    H_de = trans_to_cuda(torch.Tensor(H_de))
    time_encoder_de = trans_to_cuda(torch.Tensor(time_encoder_de))
    x_tw = trans_to_cuda(torch.Tensor(x_tw))
    H_tw = trans_to_cuda(torch.Tensor(H_tw))
    time_encoder_tw = trans_to_cuda(torch.Tensor(time_encoder_tw))
    user_label = trans_to_cuda(torch.Tensor(user_label))
    label_embedding = trans_to_cuda(torch.Tensor(label_embedding))

    output = model(x_de, H_de, time_encoder_de, x_tw, H_tw, time_encoder_tw, label_embedding)
    return output,user_label


def train_model(model, train_dataLoader, opt, label_embedding):
    model.scheduler.step()
    total_loss = 0.0
    model.train()
    for data in train_dataLoader:
        H_de, x_de, time_encoder_de, H_tw, x_tw, time_encoder_tw, user_label = data
        output, user_label = forward(model, user_label, x_de, H_de, time_encoder_de, x_tw, H_tw, time_encoder_tw,
                                     label_embedding)
        loss = model.loss_function(output, user_label)
        # 优化器优化模型
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    total_loss = total_loss / len(train_dataLoader)
    return total_loss

def test_model(model, test_dataLoader, opt, label_embedding):
    model.eval()
    num_4concat = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in test_dataLoader:
            H_de, x_de, time_encoder_de, H_tw, x_tw, time_encoder_tw, user_label = data
            output, user_label = forward(model, user_label, x_de, H_de, time_encoder_de, x_tw, H_tw, time_encoder_tw,
                                         label_embedding)
            loss = model.loss_function(output, user_label)
            total_loss += loss
            output = (output>=0.5).float()
            # for i in range(len(output)):
            #     for j in range(len(output[i])):
            #         if output[i][j] >= 0.5:
            #             output[i][j] = 1
            #         else:
            #             output[i][j] = 0
            output = trans_to_cpu(output.view(output.shape[0], output.shape[1]))  # 3x300
            user_label = trans_to_cpu(user_label.view(output.shape[0], output.shape[1]))  # 3x300
            # test_labels += list(user_label)
            # test_pred += list(output)
            if num_4concat == 0:
                test_pred = output
                test_labels = user_label
            else:
                test_pred = torch.cat((test_pred, output), 0)
                test_labels = torch.cat((test_labels, user_label), 0)
            num_4concat += 1
        total_loss = total_loss / len(test_dataLoader)
    return total_loss, test_pred, test_labels


# def train_test_model(model, train_dataLoader, test_dataLoader, opt, label_embedding):
#     model.scheduler.step()
#     # dataset = Mydata()
#     # train_dataLoader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=my_to_tensor)
#     # test_dataLoader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=my_to_tensor)
#     print('start training: ', datetime.datetime.now())
#     model.train()
#     total_loss = 0.0
#     start_time = time.time()
#     writer = SummaryWriter("logs")
#     total_train_step = 0
#     for i in range(opt.epoch):
#         print("-----第{}轮训练开始-----".format(i + 1))
#         # 训练步骤开始
#         model.train()
#         for data in train_dataLoader:
#             H_de, x_de, time_encoder_de, H_tw, x_tw, time_encoder_tw, user_label = data
#             output,user_label = forward(model, user_label, x_de, H_de, time_encoder_de, x_tw, H_tw, time_encoder_tw, label_embedding)
#             loss = model.loss_function(output, user_label)
#             # 优化器优化模型
#             model.optimizer.zero_grad()
#             loss.backward()
#             model.optimizer.step()
#             total_loss += loss
#
#             total_train_step = total_train_step + 1
#             print(total_train_step)
#             if total_train_step % 100 == 0:
#                 end_time = time.time()
#                 print(end_time - start_time)
#                 print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
#                 writer.add_scalar("train_loss", loss.item(), total_train_step)
#
#         print('\tTotal Loss:\t%.4f' % (total_loss / len(train_dataLoader)))
#
#         model.eval()
#         num_4concat = 0
#         with torch.no_grad():
#             for data in test_dataLoader:
#                 H_de, x_de, time_encoder_de, H_tw, x_tw, time_encoder_tw, user_label = data
#                 output, user_label = forward(model, user_label, x_de, H_de, time_encoder_de, x_tw, H_tw,time_encoder_tw, label_embedding)
#                 for i in range(len(output)):
#                     for j in range(len(output[i])):
#                         if output[i][j] >= 0.5:
#                             output[i][j] = 1
#                         else:
#                             output[i][j] = 0
#                 output = trans_to_cpu(output.view(opt.batchSize,label_embedding.shape[0]))#3x300
#                 user_label = trans_to_cpu(user_label.view(opt.batchSize,label_embedding.shape[0])) #3x300
#                 # test_labels += list(user_label)
#                 # test_pred += list(output)
#                 if num_4concat == 0:
#                     test_pred = output
#                     test_labels = user_label
#                 else:
#                     test_pred = torch.cat((test_pred,output),0)
#                     test_labels = torch.cat((test_labels, user_label), 0)
#                 num_4concat += 1
#
#
#             print("Macro F1-Score")
#             print(f1_score(test_labels,test_pred,average='macro'))
#             print("Micro F1-Score")
#             print(f1_score(test_labels,test_pred,average='micro'))
#             print("Example F1-Score")
#             print(f1_score(test_labels,test_pred,average='samples'))
#             print("Ranking loss")
#             print(label_ranking_loss(test_labels, test_pred))



