import torch
import torch.nn as nn
from layers.Embed_GEF import STWEmbedding
from layers.attention_GEF import STWAttBlock
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
        self.stw_embedding = STWEmbedding(self.P, self.Q, self.T, self.D, self.bn_decay)
        self.encoder_blocks = nn.ModuleList([STWAttBlock(self.Q, self.K, self.d, self.bn_decay, args.dropout) for _ in range(self.L)])
        self.projector = nn.Linear(self.D, self.Q)

    def forward(self, X, SE, TE, WE):
        #(batch,P,134,7)
        X = X.permute(0,3,2,1)
        X = self.input(X)
        sum_X = self.sum(X.permute(0,3,2,1))
        sum_X = sum_X.permute(0,3,2,1)
        #(batch,1,134,D)
        STE, TWE, SWE, STWE = self.stw_embedding(SE, TE, WE)
        for encoder_block in self.encoder_blocks:
            sum_X = encoder_block(X, sum_X, STE, SWE)
            sum_X = self.norm(sum_X)
            Y = self.dropout(self.activation(self.conv1(sum_X)))
            Y = self.dropout(self.conv2(Y))
            sum_X = self.norm(sum_X + Y)
        sum_X = self.projector(sum_X)
        #(batch,1,134,Q)
        sum_X = sum_X.permute(0,3,2,1)
        #(batch,Q,134,1)
        sum_X = torch.squeeze(sum_X, 3)
        #(batch,Q,134)
        return sum_X


