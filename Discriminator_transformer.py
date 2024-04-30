import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from newHGAT import *
from mytransformer import *
from mymultiheadattention import *
class DiscriminatorTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(DiscriminatorTransformerEncoderLayer, self).__init__()
        """
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1    
        """
        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, F0, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param src_mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return: # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """
        src2 = self.self_attn(F0, src, src,
                              key_padding_mask=src_key_padding_mask, )[0]  # 计算多头注意力
        # src2: [src_len,batch_size,num_heads*kdim] num_heads*kdim = embed_dim
        F0 = F0 + self.dropout1(src2)  # 残差连接
        F0 = self.norm1(F0)  # [src_len,batch_size,num_heads*kdim]

        src2 = self.activation(self.linear1(F0))  # [src_len,batch_size,dim_feedforward]
        src2 = self.linear2(self.dropout(src2))  # [src_len,batch_size,num_heads*kdim]
        F0 = F0 + self.dropout2(src2)
        F0 = self.norm2(F0)
        return F0  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Discriminator_TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Discriminator_TransformerEncoder, self).__init__()
        """
        encoder_layer: 就是包含有多头注意力机制的一个编码层
        num_layers: 克隆得到多个encoder layers 论文中默认为6
        norm: 归一化层
        """
        self.layers = _get_clones(encoder_layer, num_layers)
        # 克隆得到多个encoder layers 论文中默认为6
        self.num_layers = num_layers
        self.norm = norm


    def forward(self, src, F0, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:# [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """
        output = src
        F = F0
        for mod in self.layers:
            output = mod(output, F,
                         src_key_padding_mask=src_key_padding_mask)
            # 多个encoder layers层堆叠后的前向传播过程
        if self.norm is not None:
            output = self.norm(output)
        return output  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]


class DiscriminatorTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,dim_feedforward=2048, dropout=0.1):
        super(DiscriminatorTransformer, self).__init__()

        """
        :param d_model:  d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:               多头注意力机制中多头的数量，论文默认为值 8
        :param num_encoder_layers:  encoder堆叠的数量，也就是论文中的N，论文默认值为6
        :param num_decoder_layers:  decoder堆叠的数量，也就是论文中的N，论文默认值为6
        :param dim_feedforward:     全连接中向量的维度，论文默认值为 2048
        :param dropout:             丢弃率，论文中的默认值为 0.1
        """
        #  ================ 编码部分 =====================
        encoder_layer = DiscriminatorTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = Discriminator_TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters() # 初始化模型参数
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # [sz,sz]


    def forward(self, src, F0, src_key_padding_mask=None):
        """
        :param src:   [src_len,batch_size,embed_dim]
        :param tgt:  [tgt_len, batch_size, embed_dim]
        :param src_mask:  None
        :param tgt_mask:  [tgt_len, tgt_len]
        :param memory_mask: None
        :param src_key_padding_mask: [batch_size, src_len]
        :param tgt_key_padding_mask: [batch_size, tgt_len]
        :param memory_key_padding_mask:  [batch_size, src_len]
        :return: [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]
        """
        memory = self.encoder(src, F0, src_key_padding_mask=src_key_padding_mask)
        # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]

        return memory