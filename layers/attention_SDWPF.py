import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.py_utils import FC
   
class FrequencyAttention(nn.Module):
    """
    Desc:
        This class implements frequency attention mechanism that computes attention weights
        based on input features and spatial embeddings.
    Args:
        Q (int): prediction steps.
        K (int): number of attention heads.
        d (int): dims of each head attention outputs.
        bn_decay: Batch normalization decay factor.
        dropout (float): Dropout probability.
        mask (bool): Whether to apply masking in attention.
    """
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

    def forward(self, X_VMD, STE):
        """
        Desc:
            Forward pass for frequency attention. Computes the attention output based on inputs.
        Args:
            X_VMD (tensor): wind power data after VMD of shape (batch_size, 7, 134, D).
            STE (tensor): Spatial embeddings of shape (batch_size, 1, 134, D).
        Returns:
            tensor: Attention output of shape (batch_size, 1, 134, D).
        """
        batch_size = X_VMD.shape[0]
        device = X_VMD.device
        STE = STE.expand(-1,7,-1,-1)
        #(batch,7,134,D)
        X_VMD= torch.cat((X_VMD, STE), dim=-1)
        #(batch,7,134,2D)
        query = self.query_fc(X_VMD)
        key = self.key_fc(X_VMD)
        value = self.value_fc(X_VMD)
        
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key) / (self.d ** 0.5)

        if self.mask:
            num_step = X_VMD.shape[1]
            num_vertex = X_VMD.shape[2]
            mask = torch.ones(num_step, num_step, device=device)
            mask = torch.tril(mask)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)

        attention = self.dropout(F.softmax(attention, dim=-1))
        X_VMD = torch.matmul(attention, value)
        X_VMD = X_VMD.permute(0, 2, 1, 3)
        X_VMD = torch.cat(torch.split(X_VMD, batch_size , dim=0), dim=-1)
        X_VMD = self.output_fc(X_VMD)
        X_VMD = X_VMD.permute(0, 3, 2, 1)
        X_VMD = self.linear(X_VMD)
        X_VMD = X_VMD.permute(0, 3, 2, 1)
        del query, key, value, attention, batch_size
        return X_VMD
    
class FeatureAttention(nn.Module):
    """
    Desc:
        This class implements frequency attention mechanism that computes attention weights
        based on input features and spatial embeddings.
    Args:
        Q (int): prediction steps.
        K (int): number of attention heads.
        d (int): dims of each head attention outputs.
        bn_decay: Batch normalization decay factor.
        dropout (float): Dropout probability.
        mask (bool): Whether to apply masking in attention.
    """
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
        self.linear = nn.Linear(7,1)
        
    def forward(self, X, SWE):
        """
        Desc:
            Forward pass for feature attention. Computes the attention output based on inputs.
        Args:
            X (tensor): wind power after linear layer of shape (batch_size, 1, 134, D).
            SWE (tensor): Spatial-Weather embeddings of shape (batch_size, 7, 134, D).
        Returns:
            tensor: Attention output of shape (batch_size, 1, 134, D).
        """
        batch_size = X.shape[0]
        device = X.device
        #(batch,1,134,D)
        X = X.expand(-1,7,-1,-1)
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
        
class GatedFusion(nn.Module):
    """
    Desc:
        This class implements a gated fusion mechanism that combines two inputs using a gating mechanism.
    Args:
        D (int): Correlation dims
        bn_decay: Batch normalization decay factor.
    """
    def __init__(self, D, bn_decay):
        super(GatedFusion, self).__init__()
        self.D = D
        self.bn_decay = bn_decay

        self.ST_fc = FC(D, D, None, bn_decay, use_bias=False)
        self.SW_fc = FC(D, D, None, bn_decay, use_bias=True)
        self.output_fc = FC([D, D],[D,D],[F.gelu, None], bn_decay)

    def forward(self, HT, HW):
        """
        Desc:
            Forward pass for gated fusion. Combines two input features using gating mechanism.
        Args:
            HT (tensor): Input features from Frequency attention.
            HW (tensor): Input features from Feature attention.
        Returns:
            tensor: Fused output features.
        """
        ST = self.ST_fc(HT)
        SW = self.SW_fc(HW)
        z = torch.sigmoid(torch.add(ST, SW))
        H = torch.add(torch.mul(z, HT), torch.mul(1 - z, HW))
        H = self.output_fc(H)
        del ST, SW, z
        return H

class HFFEnhancement(nn.Module):
    """
    Desc:
        This class implements a Hybrid Frequency Feature Enhancement Block
    Args:
        Q (int): prediction steps.
        K (int): number of attention heads.
        d (int): dims of each head attention outputs.
        bn_decay: Batch normalization decay factor.
        dropout (float): Dropout probability.
    """
    def __init__(self, Q, K, d, bn_decay, dropout=0.1):
        
        super(HFFEnhancement, self).__init__()
        self.temporal_attention = FrequencyAttention(Q, K, d,  bn_decay, dropout, mask=False)
        self.weather_attention = FeatureAttention(Q, K, d, bn_decay, dropout, mask=False)
        self.gated_fusion = GatedFusion(K * d, bn_decay)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X_VMD, X, STE, SWE): 
        """
        Desc:
            Forward pass for Hybrid Frequency Feature Enhancement Block. Combines attention outputs.
        Args:
            X_VMD (tensor): windpower after VMD.
            X (tensor): windpower after linear layer.
            STE (tensor): Spatial embeddings.
            SWE (tensor): Spatial-Weather embeddings.
        Returns:
            tensor: Combined output features.
        """
        HT = self.temporal_attention(X_VMD, STE)
        HW = self.weather_attention(X, SWE)
        H = self.gated_fusion(HT, HW)
        Y = torch.add(X, self.dropout(H))
        del HT, HW, H
        return Y
