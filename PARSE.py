import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

import math
from math import sqrt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from collections import Counter
from sklearn import metrics

class Conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dropout = 0.5):
        super(Conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.padding = kernel_size - 1
        self.conv1d = nn.Conv1d(in_channels = self.in_dim, out_channels = self.out_dim, kernel_size = self.kernel_size, 
                                padding = self.padding)
        self.Dropout = nn.Dropout(self.dropout)
    def forward(self, data):
        data = data.permute(0, 2, 1)
        out = self.conv1d(data)[:, :, : - self.padding]
        out = out.permute(0, 2, 1)
        return self.Dropout(out)
    
class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, time_length, dim_linear, dim_k, dim_v, num_heads = 8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.time_length = time_length
        self.dim_linear = dim_linear
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear = nn.Linear(self.dim_in + self.time_length, self.dim_linear, bias = False)
        self.linear_q = nn.Linear(self.dim_linear, self.dim_k, bias=False)
        self.linear_k = nn.Linear(self.dim_linear, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_linear, self.dim_v, bias=False)
        self._norm_fact = 1 / sqrt(self.dim_k // self.num_heads)

    def forward(self, data):
        # x: tensor of shape (batch, n, dim_in)
        batch_size, time_length, dim_in = data.shape
        assert dim_in == self.dim_in and time_length == self.time_length
        p = torch.stack([torch.eye(time_length)] * batch_size, dim = 0).cuda()
#         p = torch.stack([torch.eye(time_length)] * batch_size, dim = 0)
        data = self.linear(torch.cat([data, p], dim = -1))
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(data).reshape(batch_size, time_length, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(data).reshape(batch_size, time_length, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(data).reshape(batch_size, time_length, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim = -1)  # batch, nh, n, n
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch_size, time_length, self.dim_v)  # batch, n, dim_v
        return att
    
class RecalibrationModule(nn.Module):
    def __init__(self, in_dim, compress_ratio = 8):
        super(RecalibrationModule, self).__init__()
        self.in_dim = in_dim
        self.compress_ratio = compress_ratio      
        self.linear_dim = int(self.in_dim / self.compress_ratio)
        self.W = nn.Linear(self.in_dim, self.linear_dim)
        self.U = nn.Linear(self.linear_dim, self.in_dim)
    def forward(self, data):
        data = data.float()
        return data * torch.sigmoid(self.U(torch.relu(self.W(data))))
    

class PARSE(nn.Module):
    def __init__(self, in_dim, feature_num, hidden_dim, self_attention_dim, guide_attention_dim, out_dim = 1, dropout = 0.5):
        super(PARSE, self).__init__()
        self.in_dim = in_dim
        self.feature_num = feature_num
        self.hidden_dim = hidden_dim
        self.self_attention_dim = self_attention_dim
        self.guide_attention_dim = guide_attention_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        self.kernel_size = [3, 4, 5]
        self.normal = torch.Tensor([0, 59, 0.21, 4, 6, 15, 5, 128, 86, 170, 77, 98, 19, 118, 36.6, 81, 7.4]).cuda()
        self.gru = nn.GRU(input_size = self.in_dim, hidden_size = self.hidden_dim, num_layers = 2, 
                          batch_first = True, dropout = self.dropout)
        
        self.conv1d_a = nn.ModuleList([
            nn.Sequential(
                Conv(in_dim = self.feature_num, out_dim = self.hidden_dim, kernel_size = h, dropout = self.dropout),
                nn.Tanh()
            )
            for h in self.kernel_size
        ])
        self.conv1d_r = nn.ModuleList([
            nn.Sequential(
                Conv(in_dim = self.feature_num, out_dim = self.hidden_dim, kernel_size = h, dropout = self.dropout),
                nn.Tanh()
            )
            for h in self.kernel_size
        ])
        
        self.guide_linear_a = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.guide_attention_dim),
            nn.Tanh()
        )
        self.guide_V_a = nn.Linear(self.guide_attention_dim, 1, bias = False)
        self.guide_linear_r = nn.Sequential(
            nn.Linear(3 * self.hidden_dim + self.self_attention_dim, self.guide_attention_dim),
            nn.Tanh()
        )
        self.guide_V_a = nn.Linear(self.guide_attention_dim, 1, bias = False)
        
        self.query_linear_a = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.query_linear_r = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        
        self.rm_a = nn.ModuleList([
            RecalibrationModule(self.hidden_dim * 3)
        ] * 48)
        self.rm_r = nn.ModuleList([
            RecalibrationModule(self.hidden_dim * 3)
        ] * 48)
        self.rm_data = nn.ModuleList([
            RecalibrationModule(self.in_dim)
        ] * 48)

        self.lstm = nn.LSTM(input_size = 3 * self.hidden_dim + self.in_dim, hidden_size = self.self_attention_dim, num_layers = 2,  batch_first = True, dropout = self.dropout)
        
        
        
        self.MultiHeadSelfAttention = MultiHeadSelfAttention(self.hidden_dim * 4, 48, self.hidden_dim, self.self_attention_dim, 
                                                             self.self_attention_dim)
        
        self.get_alpha = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, 1, bias = False),
            nn.Sigmoid()
        )
    
        
        self.get_beta = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.output_classifier = nn.Sequential(
            nn.Linear(self.self_attention_dim, self.out_dim, bias = False),
            nn.Sigmoid()
        )
        self.Dropout1 = nn.Dropout(self.dropout)
        self.Dropout2 = nn.Dropout(self.dropout)

    def get_offset(self, data):
        bias = 1e-4
        batch_size, time, feature_num = data.shape
        data2 = data - self.normal
        data2[:, :, 1:] = data2[:, :, 1:] / self.normal[1:]#除了第一个特征之外，除以正常值，第一个特征是0或1
        absolute = torch.tanh(data2)
        zeros = torch.zeros(batch_size, 1, feature_num).cuda()
        pad = torch.cat([zeros, data], dim = 1)[:, : -1, :]#往后移动一个时间步，第一个时间步补零
        data3 = (data - pad)[:, 1:, :]#相减得到（time - 1）个差值
        data3[:, :, 1:] = data3[:, :, 1:] / (pad[:, 1:, 1:] + bias)#除了第一个特征之外，除以前一时间步的值，第一个特征是0或1
        relative = torch.tanh(torch.cat([data3, zeros], dim = 1))#第一个时间步补零
        return absolute, relative
    
    def forward(self, data, data_nn):
        data = data.float().cuda()
        data_nn = data_nn.float().cuda()
        gru_output, _ = self.gru(data)
        gru_final_output = gru_output[:, -1, :]
        
        a, r = self.get_deviation(data_nn)
        
        a = a.cuda()
        r = r.cuda()
        
        a_conv = torch.cat([conv(a) for conv in self.conv1d_a], dim = -1)
        a_query_attention_input = self.query_linear_a(a_conv).matmul(gru_final_output.unsqueeze(2))
        a_query_attention_weight = torch.softmax(a_query_attention_input, dim = 1)
        a_query_attention_result = torch.sum(a_query_attention_weight * a_conv, dim = 1) / math.sqrt(self.hidden_dim * 3)

        r_conv = torch.cat([conv(r) for conv in self.conv1d_r], dim = -1)
        r_query_attention_input = self.query_linear_d(r_conv).matmul(gru_final_output.unsqueeze(2))
        r_query_attention_weight = torch.softmax(r_query_attention_input, dim = 1)
        r_query_attention_result = torch.sum(r_query_attention_weight * r_conv, dim = 1) / math.sqrt(self.hidden_dim * 3)
        
        a_rm = torch.stack([self.rm_a[i](a_conv[:, i, :]) for i in range(48)], dim = 1)
        data_rm = torch.stack([self.rm_data[i](data[:, i, :]) for i in range(48)], dim = 1)
        a_lstm_input = torch.cat([a_rm, data_rm], dim = -1)
        a_lstm_output, _ = self.lstm(a_lstm_input)
        a_lstm_result = a_lstm_output[:, -1, :]

        
        r_rm = torch.stack([self.rm_r[i](r_conv[:, i, :]) for i in range(48)], dim = 1)
        r_self_attention_input = torch.cat([r_rm, gru_output], dim = -1)
        r_self_attention_result = self.Dropout1(self.MultiHeadSelfAttention(r_self_attention_input))
        r_self_attention_result = r_self_attention_result.mean(1)

        alpha = self.get_alpha(a_query_attention_result)
        beta = self.get_beta(r_query_attention_result)
        result = (alpha / (alpha + beta)) * a_lstm_result + (beta / (alpha + beta)) * r_self_attention_result
        result = self.Dropout2(result)
        
        return self.output_classifier(result)