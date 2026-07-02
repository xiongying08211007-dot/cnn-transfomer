import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, variable
import math
import matplotlib.pylab as plt
import numpy as np
import copy

from torch.nn.functional import dropout


class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
#d_model：词嵌入维度
#vocab:词表大小
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model=d_model
    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)


d_model = 512
vocab = 1000

x=Variable(torch.LongTensor([[100,2,21,508],[491,998,1,211]]))

emb = Embeddings(d_model,vocab)
embr = emb(x)
print("embr:",embr)
print(embr.shape)




class PositionEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #初始化位置编码
        pe = torch.zeros(max_len,d_model)

        #初始化绝对位置编码
        position = torch.arange(0,max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)


        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)


    def forward(self,x):
        x = x + self.pe[:, :x.size(1)].clone().detach().to(x.device)

        return self.dropout(x)



d_model=512
dropout=0.1
max_len =60

x=embr
pe=PositionEncoding(d_model,dropout,max_len)
pe_result = pe(x)
print(pe_result)
print(pe_result.shape)