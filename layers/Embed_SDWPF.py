import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Spatial Embedding class
class SpatialEmbedding(nn.Module):
    """
    Desc:
        This class creates spatial embeddings by applying a linear transformation to the spatial data.
    Args:
        P (int): history steps.
        D (int): Correlation dimension.
    """
    def __init__(self, P, D):
        super(SpatialEmbedding, self).__init__()
        self.linear = nn.Linear(P, D, bias=False)
    def forward(self, SE):
        """
        Desc:
            Forward pass for spatial embedding. Expands the input tensor, applies linear transformation, and returns the result.
        Args:
            SE (tensor): Input spatial data of shape (batch_size, 134, P).
        Returns:
            tensor: Embedded spatial data of shape (batch_size, 1, 134, D).
        """
        SE = SE.unsqueeze(1)
        SE = self.linear(SE)
        return SE

# Define the Temporal Embedding class
class TemporalEmbedding(nn.Module):
    """
    Desc:
        This class creates temporal embeddings using one-hot encodings of time-related data (month, day of the week, and time of day).
    Args:
        P (int): history steps.
        T (int): Time of day granularity (number of time slots per day).
        D (int): Correlation dimension.
    """
    def __init__(self, P, T, D):
        super(TemporalEmbedding, self).__init__()
        self.T = T
        self.linear_time = nn.Linear(163, 1, bias=False)
        self.linear = nn.Linear(P, D, bias=False)

    def forward(self, TE):
        """
        Desc:
            Forward pass for temporal embedding. Uses one-hot encoding for month, day of the week, and time of day.
        Args:
            TE (tensor): Input temporal data of shape (batch_size, P, 3) where 3 represents [month, day_of_week, time_of_day].
        Returns:
            tensor: Embedded temporal data of shape (batch_size, 1, 1, D).
        """
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
        #(batch_size,163,1,D)
        TE = TE.permute(0,3,2,1)
        TE = self.linear_time(TE)
        TE = TE.permute(0,3,2,1)
        return TE

# Define the Weather Embedding class
class WeatherEmbedding(nn.Module):
    """
    Desc:
        This class creates weather embeddings by applying a linear transformation to the weather data.
    Args:
        P (int): history steps.
        D (int): Correlation dimension.
    """
    def __init__(self, P, D):
        super(WeatherEmbedding, self).__init__()
        self.linear = nn.Linear(P, D, bias=False)
      
    def forward(self, WE):
        """
        Desc:
            Forward pass for weather embedding. Rearranges and embeds the weather data.
        Args:
            WE (tensor): Input weather data of shape (batch_size, P, 134, 7).
        Returns:
            tensor: Embedded weather data of shape (batch_size, 4, 134, D).
        """
        WE = WE.permute(0,3,2,1)
        WE = self.linear(WE)
        return WE

# Define the Combined Embedding class (Spatial, Temporal, Weather Embedding)
class Embedding(nn.Module):
    """
    Desc:
        This class combines spatial, temporal, and weather embeddings. Each type of embedding is handled by its respective class.
    Args:
        P (int): history steps.
        T (int): Time of day granularity (number of time slots per day).
        D (int): Correlation dimension.
    """
    def __init__(self, P, T, D):
        super(Embedding, self).__init__()
        self.spatial_embedding = SpatialEmbedding(P, D)
        self.temporal_embedding = TemporalEmbedding(P, T, D)
        self.weather_embedding = WeatherEmbedding(P, D)

    def forward(self, SE, TE, WE):
        """
        Desc:
            Forward pass for the combined embedding class. Computes spatial, temporal, and weather embeddings.
        Args:
            SE (tensor): Spatial embedding input of shape (batch_size, 134, P).
            TE (tensor): Temporal embedding input of shape (batch_size, P, 3).
            WE (tensor): Weather embedding input of shape (batch_size, P, 134, 7).
        Returns:
            tuple: Contains the different combinations of embeddings:
                   - STE: Spatial embedding and temporal embedding combined. In Inverted transformer don't need temporal embedding.
                   - TWE: Temporal and weather embedding combined.
                   - SWE: Spatial and weather embedding combined.
                   - STWE: Spatial, temporal, and weather embedding combined.
        """
        SE_embed = self.spatial_embedding(SE)
        TE_embed = self.temporal_embedding(TE)
        WE_embed = self.weather_embedding(WE)

        SE = SE_embed 
        #STE = SE_embed + TE_embed
        #(batch_size,1,134,D)
        TWE = TE_embed + WE_embed
        #(batch_size,7,134,D)
        SWE = SE_embed + WE_embed
        #(batch_size,7,134,D)
        STWE = SE_embed + TE_embed + WE_embed
        #(batch_size,7,134,D)

        return SE, TWE, SWE, STWE


