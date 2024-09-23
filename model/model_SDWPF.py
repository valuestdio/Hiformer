import torch
import torch.nn as nn
from layers.Embed_SDWPF import Embedding
from layers.attention_SDWPF import HFFEnhancement
import torch.nn.functional as F
from utils.py_utils import FC

class Hiformer(nn.Module):
    def __init__(self, args, bn_decay):
        super(Hiformer, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.L = args.L
        self.K = args.K
        self.d = args.d
        self.T = args.T
        self.d_ff = args.d_ff
        self.D = self.K * self.d
        self.bn_decay = bn_decay
        self.input = nn.Linear(self.P, self.D)
        self.sum = nn.Linear(7, 1)
        self.norm = nn.LayerNorm(self.D)
        self.conv1 = FC(self.D, self.d_ff, F.gelu,  bn_decay)
        self.conv2 = FC(self.d_ff, self.D, F.gelu,  bn_decay)
        self.dropout = nn.Dropout(args.dropout)
        self.activation = F.gelu
        self.stw_embedding = Embedding(self.P, self.T, self.D)
        self.encoder_blocks = nn.ModuleList([HFFEnhancement(self.Q, self.K, self.d, self.bn_decay, args.dropout) for _ in range(self.L)])
        self.projector = nn.Linear(self.D, self.Q)

    def forward(self, X_VMD, SE, TE, WE):
        #(batch,P,134,7)
        X_VMD = X_VMD.permute(0,3,2,1)
        X_VMD = self.input(X_VMD)
        X = self.sum(X_VMD.permute(0,3,2,1))
        X = X.permute(0,3,2,1)
        #(batch,1,134,D)
        STE, TWE, SWE, STWE = self.stw_embedding(SE, TE, WE)
        for encoder_block in self.encoder_blocks:
            X = encoder_block(X_VMD, X, STE, SWE)
            X = self.norm(X)
            Y = self.dropout(self.activation(self.conv1(X)))
            Y = self.dropout(self.conv2(Y))
            X = self.norm(X + Y)
        X = self.projector(X)
        #(batch,1,134,Q)
        X = X.permute(0,3,2,1)
        #(batch,Q,134,1)
        X = torch.squeeze(X, 3)
        #(batch,Q,134)
        return X


