import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# 定义批量归一化函数
class BatchNorm(nn.Module):
    def __init__(self, num_features, bn_decay):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, momentum=1 - bn_decay)

    def forward(self, x):
        return self.bn(x)

# 定义卷积层函数
class Conv2dLayer(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1), padding='same', use_bias=True, activation=F.relu, bn_decay=None):
        super(Conv2dLayer, self).__init__()
        self.activation = activation  # Use the function directly
        if padding == 'same':
            padding_size = (math.ceil((kernel_size - 1) / 2), math.ceil((kernel_size - 1) / 2))
        elif padding == 'VALID':
            padding_size = (0, 0)
        else:
            raise ValueError("Padding must be 'same' or 'valid'")

        self.padding_size = padding_size
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride, padding=0, bias=use_bias)
        self.bn = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        #need input data structure(N,C,H,W) batch size, channels, height, width
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, (self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]))
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x.permute(0, 3, 2, 1)

# 定义全连接层函数
class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([Conv2dLayer(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
    
def mae_loss(pred, label):
    mask = (label != 0).float()
    mask /= mask.mean()
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0, device=mask.device), mask)
    
    loss = torch.abs(pred - label)
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0, device=loss.device), loss)
    
    return loss.mean()

class AddNorm(nn.Module):
    def __init__(self, feature_dim, epsilon=1e-6):
        super(AddNorm, self).__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(feature_dim))

    def forward(self, X, Y):
        residual = X
        mean = Y.mean(dim=-1, keepdim=True)
        variance = Y.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (Y - mean) / torch.sqrt(variance + self.epsilon)
        outputs = self.scale * normalized + residual
        return outputs