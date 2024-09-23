import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BatchNorm(nn.Module):
    """
    Desc:
        Batch Normalization layer to normalize the input batch of 2D feature maps.
    Args:
        num_features: The number of feature maps in the input.
        bn_decay: Decay rate for the moving averages of mean and variance.
    Returns:
        Normalized feature maps.
    """
    def __init__(self, num_features, bn_decay):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, momentum=1 - bn_decay)

    def forward(self, x):
        """
        Desc:
            Forward pass of the batch normalization.
        Args:
            x: Input tensor of shape (batch_size, num_features, height, width).
        Returns:
            Normalized output tensor.
        """
        return self.bn(x)

# 定义卷积层函数
class Conv2dLayer(nn.Module):
    """
    Desc:
        2D Convolutional Layer with optional batch normalization and activation.
    Args:
        input_dims: Number of input channels.
        output_dims: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        padding: Type of padding ('same' or 'VALID').
        use_bias: Whether to use bias in the convolution.
        activation: Activation function to apply after convolution.
        bn_decay: Decay rate for the batch normalization.
    Returns:
        Convolved and normalized feature maps.
    """
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
        """
        Desc:
            Forward pass of the convolutional layer.
        Args:
            x: Input tensor of shape (batch_size, width, height , channels).
        Returns:
            Convolved, normalized, and activated output tensor.
        """
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
    """
    Desc:
        Fully Connected Layer composed of multiple convolutional layers.
    Args:
        input_dims: Input dimensions for each layer.
        units: Number of units (output dimensions) for each layer.
        activations: Activation functions for each layer.
        bn_decay: Decay rate for the batch normalization.
        use_bias: Whether to use bias in the convolution.
    Returns:
        Output of the fully connected network.
    """
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
        """
        Desc:
            Forward pass of the fully connected layers.
        Args:
            x: Input tensor.
        Returns:
            Output tensor after passing through all fully connected layers.
        """
        for conv in self.convs:
            x = conv(x)
        return x
    
def mae_loss(pred, label):
    """
    Desc:
        Mean Absolute Error (MAE) loss with masking.
    Args:
        pred: Predicted values.
        label: Ground truth values.
    Returns:
        Mean Absolute Error value.
    """
    mask = (label != 0).float()
    mask /= mask.mean()
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0, device=mask.device), mask)
    
    loss = torch.abs(pred - label)
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0, device=loss.device), loss)
    
    return loss.mean()

class AddNorm(nn.Module):
    """
    Desc:
        Add & Normalize Layer for residual connections.
    Args:
        feature_dim: Dimensionality of the input features.
        epsilon: Small constant to avoid division by zero during normalization.
    Returns:
        Normalized output with residual connection.
    """
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