import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.py_utils import FC

# 定义空间嵌入类
class SpatialEmbedding(nn.Module):
    def __init__(self, P, Q, D,  bn_decay):
        super(SpatialEmbedding, self).__init__()
        self.linear = nn.Linear(P, D, bias=False)
    def forward(self, SE):
        #shape(batch_size,134,P)
        SE = SE.unsqueeze(1)
        #SE = self.fc(SE)
        SE = self.linear(SE)
        #(batch_size,1,134,D)
        return SE

# 定义时间嵌入类
class TemporalEmbedding(nn.Module):
    def __init__(self, P, Q, T, D, bn_decay):
        super(TemporalEmbedding, self).__init__()
        self.T = T
        self.linear_time = nn.Linear(163, 1, bias=False)
        self.linear = nn.Linear(P, D, bias=False)

    def forward(self, TE):
        #(batch_size,P,3)
        TE = TE.long()  # Ensure TE is of integer type
        month =  torch.empty(TE.shape[0], TE.shape[1], 12,device=TE.device)
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7,device=TE.device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], self.T,device=TE.device)

        month = F.one_hot(TE[..., 0] % 12, num_classes=12).to(TE.device)
        dayofweek = F.one_hot(TE[..., 1] % 7, num_classes=7).to(TE.device)
        timeofday = F.one_hot(TE[..., 2] % self.T, num_classes=self.T).to(TE.device)

        TE = torch.cat((month, dayofweek, timeofday), dim=-1).float()
        TE = TE.unsqueeze(2)
        TE = TE.permute(0,3,2,1)
        #(batch_size,163,1,P)
        TE = self.linear(TE)
        #TE = self.fc(TE)
        #(batch_size,163,1,D)
        TE = TE.permute(0,3,2,1)
        TE = self.linear_time(TE)
        TE = TE.permute(0,3,2,1)
        return TE

# 定义天气嵌入类
class WeatherEmbedding(nn.Module):
    def __init__(self, P, Q, D,  bn_decay):
        super(WeatherEmbedding, self).__init__()
        self.linear = nn.Linear(P, D, bias=False)
      
    def forward(self, WE):
        #(batch_size,P,134,7)
        WE = WE.permute(0,3,2,1)
        #(batch_size,7,134,P)
        WE = self.linear(WE)
        #WE = self.fc(WE)
        #(batch_size,7,134,D)
        return WE

# 定义组合嵌入类
class STWEmbedding(nn.Module):
    def __init__(self, P, Q, T, D, bn_decay):
        super(STWEmbedding, self).__init__()
        self.spatial_embedding = SpatialEmbedding(P, Q, D, bn_decay)
        self.temporal_embedding = TemporalEmbedding(P, Q, T, D, bn_decay)
        self.weather_embedding = WeatherEmbedding(P, Q, D, bn_decay)

    def forward(self, SE, TE, WE):
        SE_embed = self.spatial_embedding(SE)
        TE_embed = self.temporal_embedding(TE)
        WE_embed = self.weather_embedding(WE)

        SE = SE_embed 
        #(batch_size,1,134,D)
        TWE = TE_embed + WE_embed
        #(batch_size,7,134,D)
        SWE = SE_embed + WE_embed
        #(batch_size,7,134,D)
        STWE = SE_embed + TE_embed + WE_embed
        #(batch_size,7,134,D)

        return SE, TWE, SWE, STWE


