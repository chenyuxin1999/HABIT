import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from Discriminator_transformer import *

class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, nhead, num_encoder_layers, dim_feedforward, dropout=0.3):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.transformer_de = DiscriminatorTransformer(d_model=output_size, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward)
        self.transformer_tw = DiscriminatorTransformer(d_model=output_size, nhead=nhead,
                                                       num_encoder_layers=num_encoder_layers,
                                                       dim_feedforward=dim_feedforward)
        self.F0linear = nn.Linear(input_size, output_size)
        self.catlinear = nn.Linear(2 * output_size, output_size)
        self.outlinear = nn.Linear(output_size, 1)



    def forward(self, de_out, tw_out, F0):
        print(de_out.shape)
        print(tw_out.shape)
        batchsize = de_out.shape[1]
        F0 = F0.repeat(batchsize,1,1) #5x300x100
        F0 = F0.transpose(0,1)  #300x5x100
        F0 = self.F0linear(F0) # 300x5x60
        F = torch.tanh(F0).transpose(0,1) # 5x300x60

        de_out = self.transformer_de(src=de_out, F0=F0, src_key_padding_mask=None) # 300x5x60
        tw_out = self.transformer_tw(src=tw_out, F0=F0, src_key_padding_mask=None) # 300x5x60

        concat = torch.cat((de_out,tw_out),2) #300x5x120
        concat = self.catlinear(concat).transpose(0,1) # 5x300x60
        concat = torch.tanh(concat)
        out = concat - F # 5x300x60
        out = out * out  # 5x300x60
        out_lin = self.outlinear(out)
        out = torch.sigmoid(out_lin)
        # for i in range(len(out)):
        #     for j in range(len(out[i])):
        #         if out[i][j] >= 0.5:
        #             out[i][j] = 1
        #         else:
        #             out[i][j] = 0
        return out #5x300x1


