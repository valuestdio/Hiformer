import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.py_utils import FC
   
class FrequencyAttention(nn.Module):
    def __init__(self, Q, K, d, bn_decay, dropout=0.1, mask=True):
        super(FrequencyAttention, self).__init__()
        self.Q = Q
        self.K = K
        self.d = d
        self.D = K * d
        self.bn_decay = bn_decay
        self.mask = mask
        self.dropout = nn.Dropout(dropout)
        self.query_fc = FC(2*self.D, self.D, F.gelu,  bn_decay)
        self.key_fc = FC(2*self.D, self.D, F.gelu,  bn_decay)
        self.value_fc = FC(2*self.D, self.D, F.gelu,  bn_decay)
        self.output_fc = FC(self.D, self.D, F.gelu,   bn_decay)
        self.linear = nn.Linear(7,1)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        device = X.device
        STE = STE.expand(-1,7,-1,-1)
        #(batch,7,134,D)
        X = torch.cat((X, STE), dim=-1)
        #(batch,7,134,2D)
        query = self.query_fc(X)
        key = self.key_fc(X)
        value = self.value_fc(X)
        
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key) / (self.d ** 0.5)

        if self.mask:
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step, device=device)
            mask = torch.tril(mask)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)

        attention = self.dropout(F.softmax(attention, dim=-1))
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size , dim=0), dim=-1)
        X = self.output_fc(X)
        X = X.permute(0, 3, 2, 1)
        X = self.linear(X)
        X = X.permute(0, 3, 2, 1)
        del query, key, value, attention, batch_size
        return X
    
class FeatureAttention(nn.Module):
    def __init__(self, Q, K, d, bn_decay, dropout=0.1, mask=True):
        super(FeatureAttention, self).__init__()
        self.Q = Q
        self.K = K
        self.d = d
        self.D = K * d
        self.bn_decay = bn_decay
        self.mask = mask
        self.dropout = nn.Dropout(dropout)
        self.query_fc = FC(2*self.D, self.D, F.gelu,  bn_decay)
        self.key_fc = FC(2*self.D, self.D, F.gelu,  bn_decay)
        self.value_fc = FC(2*self.D, self.D, F.gelu,  bn_decay)
        self.output_fc = FC(self.D, self.D, F.gelu,   bn_decay)
        self.linear = nn.Linear(4,1)
        
    def forward(self, X, SWE):
        batch_size = X.shape[0]
        device = X.device
        #(batch,1,134,D)
        X = X.expand(-1,4,-1,-1)
        X = torch.cat((X, SWE), dim=-1)
        #(batch,7,134,2D)
        query = self.query_fc(X)
        key = self.key_fc(X)
        value = self.value_fc(X)
        
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key) / (self.d ** 0.5)

        if self.mask:
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step, device=device)
            mask = torch.tril(mask)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)

        attention = self.dropout(F.softmax(attention, dim=-1))
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size , dim=0), dim=-1)
        X = self.output_fc(X)
        X = X.permute(0, 3, 2, 1)
        X = self.linear(X)
        X = X.permute(0, 3, 2, 1)
        del query, key, value, attention, batch_size
        return X
        
# 定义 gatedFusion 类
class GatedFusion(nn.Module):
    def __init__(self, Q, D, bn_decay):
        super(GatedFusion, self).__init__()
        self.D = D
        self.bn_decay = bn_decay

        self.XS_fc = FC(D, D, None, bn_decay, use_bias=False)
        self.XT_fc = FC(D, D, None, bn_decay, use_bias=True)
        self.output_fc = FC([D, D],[D,D],[F.gelu, None], bn_decay)

    def forward(self, HS, HT):
        XS = self.XS_fc(HS)
        XT = self.XT_fc(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.output_fc(H)
        del XS, XT, z
        return H

class STWAttBlock(nn.Module):
    def __init__(self, Q, K, d, bn_decay, dropout=0.1):
        super(STWAttBlock, self).__init__()
        self.temporal_attention = FrequencyAttention(Q, K, d,  bn_decay, dropout, mask=False)
        self.weather_attention = FeatureAttention(Q, K, d, bn_decay, dropout, mask=False)
        self.gated_fusion = GatedFusion(Q, K * d, bn_decay)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, sum_X, STE, SWE): 
        HT = self.temporal_attention(X, STE)
        HW = self.weather_attention(sum_X, SWE)
        H = self.gated_fusion(HT, HW)
        Y = torch.add(sum_X, self.dropout(H))
        del HT, HW, H
        return Y
