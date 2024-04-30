import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

class HyperGraphAttentionLayerSparse(nn.Module):
    def __init__(self,in_features, out_features,alpha=0.2):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.weight1 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight4 = Parameter(torch.Tensor(self.in_features, self.out_features))

        self.w3 = Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.w5 = Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight4.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.w3.data, -stdv, stdv)
        nn.init.uniform_(self.w5.data, -stdv, stdv)


    def forward(self, word, adj):

        batchsize = adj.shape[0]
        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[2]  # number of node

        adj_4mp = torch.clone(adj)
        tmp_adj_4mp = torch.sum(adj_4mp, dim=-1, keepdim=True)
        tmp_adj_4mp = torch.where(tmp_adj_4mp > 0, tmp_adj_4mp, torch.ones_like(tmp_adj_4mp))
        adj_4mp = torch.div(adj_4mp, tmp_adj_4mp)
        # for i in range(batchsize):  # 第i个用户
        #     for j in range(N1):  # 第j条边
        #         if len(adj[i][j].nonzero()) != 0:
        #             adj_4mp[i][j] = adj[i][j] / len(adj[i][j].nonzero())
        # # batch中所有用户每条边对应的h_mp，以de为例子，word为5x855x60,adj为5x100x855,h_mp为5x100x60
        mp = torch.matmul(adj_4mp, word)

        #h_mp 5x100x855x60，adj不为0的位置放的是对应边的节点meanpooling
        # h_mp = trans_to_cuda(torch.zeros((batchsize,N1,N2,self.in_features)))
        # index = adj.nonzero()
        # h1 = torch.zeros_like(h_mp)  
        # for i,j,k in index:
        #     h_mp[i][j][k] = mp[i][j]
        #     h1[i][j][k] = word[i][k]

        mp = mp.unsqueeze(dim=2)  # 5x100x1x60
        mp = mp.expand(batchsize, N1, N2, self.in_features)  # 5x100x855x60
        adj_4mp = torch.clone(adj)
        adj_4mp = adj_4mp.unsqueeze(dim=3)  # 5x100x855x1
        h_mp = mp.mul(adj_4mp)  # 5x100x855x60
        h1 = torch.clone(word)
        h1 = h1.unsqueeze(dim=2)  # 5x855x1x60
        h1 = h1.expand(batchsize, N2, N1, self.in_features)  # 5x855x100x60
        adj_4mp = adj_4mp.transpose(1, 2)  # 5x855x100x1
        h1 = h1.mul(adj_4mp)  # 5x855x100x60
        h1 = h1.transpose(1, 2)  # 5x855x100x60

        w2h = h_mp.matmul(self.weight2) #5x100x855x60
        w1h = h1.matmul(self.weight1) #5x100x855x60
        concat = torch.cat((w1h,w2h),3)#5x100x855x120
        concat = concat.matmul(self.w3)#5x100x855x1
        attention_alpha = torch.exp(self.leakyrelu(concat)).view(batchsize,N1,N2)#5x100x855
        zero_vec = -9e15 * torch.ones_like(attention_alpha)
        attention = torch.where(adj > 0, attention_alpha, zero_vec)
        attention_edge = F.softmax(attention, dim=2)#5x100x855
        edge = torch.matmul(attention_edge, word)#5x100x60

        adj_t = adj.transpose(1, 2)
        # h = torch.zeros_like(h_mp.transpose(1, 2))  # 5X855X100X60
        # e = torch.zeros_like(h_mp.transpose(1,2))#5X855X100X60
        # index_h = adj_t.nonzero()
        # for p,q,r in index_h:
        #     h[p][q][r] = word[p][q]
        #     e[p][q][r] = edge[p][r]
        word2 = word.unsqueeze(dim=2)  # 5x855x1x60
        word2 = word2.expand(batchsize, N2, N1, self.in_features)  # 5x855x100x60
        adj_4mp = torch.clone(adj_t)
        adj_4mp = adj_4mp.unsqueeze(dim=3)  # 5x855x100x1
        h = word2.mul(adj_4mp)  # 5x855x100x60
        h = trans_to_cuda(h)

        e = torch.clone(edge)#5x100x60
        e = e.unsqueeze(dim=2)  # 5x100x1x60
        e = e.expand(batchsize, N1, N2, self.in_features)  # 5x100x855x60
        adj_4mp = adj_4mp.transpose(1, 2)  # 5x100x855x1
        e = e.mul(adj_4mp)  # 5x100x855x60
        e = e.transpose(1, 2)  # 5x855x100x60
        e = trans_to_cuda(e)



        w1h = h.matmul(self.weight1)#5X855X100X60
        w4e = e.matmul(self.weight4)#5X855X100X60
        concat = torch.cat((w1h,w4e),3)#5X855X100X120
        concat = concat.matmul(self.w5)  # 5x855X100x1
        attention_beta = torch.exp(self.leakyrelu(concat)).view(batchsize, N2, N1)  # 5x855X100
        zero_vec = -9e15 * torch.ones_like(attention_beta)
        attention = torch.where(adj_t > 0, attention_beta, zero_vec)
        attention_node = F.softmax(attention, dim=2)  # 5x855x100
        node = torch.matmul(attention_node, edge)  # 5x855x60
        return node + word



class HyperGraphAttentionOutLayer(nn.Module):
    def __init__(self,in_features, out_features, alpha=0.2):
        super(HyperGraphAttentionOutLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.weight1 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))

        self.w3 = Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.w3.data, -stdv, stdv)

    def forward(self, word, adj):
        batchsize = adj.shape[0]
        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[2]  # number of node

        adj_4mp = torch.clone(adj)
        tmp_adj_4mp = torch.sum(adj_4mp, dim=-1, keepdim=True)
        tmp_adj_4mp = torch.where(tmp_adj_4mp > 0, tmp_adj_4mp, torch.ones_like(tmp_adj_4mp))
        adj_4mp = torch.div(adj_4mp, tmp_adj_4mp)
        # batch中所有用户每条边对应的h_mp，以de为例子，word为5x855x60,adj为5x100x855,h_mp为5x100x60
        mp = torch.matmul(adj_4mp, word)

        # h_mp 5x100x855x60，adj不为0的位置放的是对应边的节点meanpooling
        # h_mp = trans_to_cuda(torch.zeros((batchsize, N1, N2, self.in_features)))
        # index = adj.nonzero()
        # h1 = torch.zeros_like(h_mp)
        # for i, j, k in index:
        #     h_mp[i][j][k] = mp[i][j]
        #     h1[i][j][k] = word[i][k]

        mp = mp.unsqueeze(dim=2)  # 5x100x1x60
        mp = mp.expand(batchsize, N1, N2, self.in_features)  # 5x100x855x60
        adj_4mp = torch.clone(adj)
        adj_4mp = adj_4mp.unsqueeze(dim=3)  # 5x100x855x1
        h_mp = mp.mul(adj_4mp)  # 5x100x855x60
        h1 = torch.clone(word)
        h1 = h1.unsqueeze(dim=2)  # 5x855x1x60
        h1 = h1.expand(batchsize, N2, N1, self.in_features)  # 5x855x100x60
        adj_4mp = adj_4mp.transpose(1, 2)  # 5x855x100x1
        h1 = h1.mul(adj_4mp)  # 5x855x100x60
        h1 = h1.transpose(1, 2)  # 5x855x100x60


        w2h = h_mp.matmul(self.weight2)  # 5x100x855x60
        w1h = h1.matmul(self.weight1)  # 5x100x855x60
        concat = torch.cat((w1h, w2h), 3)  # 5x100x855x120
        concat = concat.matmul(self.w3)  # 5x100x855x1
        attention_alpha = torch.exp(self.leakyrelu(concat)).view(batchsize, N1, N2)  # 5x100x855
        zero_vec = -9e15 * torch.ones_like(attention_alpha)
        attention = torch.where(adj > 0, attention_alpha, zero_vec)
        attention_edge = F.softmax(attention, dim=2)  # 5x100x855
        edge = torch.matmul(attention_edge, word)  # 5x100x60
        return edge




