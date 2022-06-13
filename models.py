import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GATEConv
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import global_mean_pool

class GATE(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super().__init__()

        self.conv1 = GATEConv(in_channels, config.hidden_channels_1, config.att_channels, edge_dim=1, heads=config.heads, dropout=config.dropout)
        self.conv2 = GATEConv(config.hidden_channels_1, config.hidden_channels_2, config.att_channels, edge_dim=1, heads=config.heads, dropout=config.dropout)
        self.conv3 = GATEConv(config.hidden_channels_2, config.hidden_channels_3, config.att_channels, edge_dim=1, heads=config.heads, dropout=config.dropout)

        self.bn1 = nn.BatchNorm1d(config.hidden_channels_1)
        self.bn2 = nn.BatchNorm1d(config.hidden_channels_2)
        self.bn3 = nn.BatchNorm1d(config.hidden_channels_3)

        self.linear = nn.Linear(config.hidden_channels_3, out_channels)

    def forward(self, data):
        x, edge_index, weight, batch = data.x, data.edge_index, data.weight, data.batch
        edge_attr = weight.unsqueeze(dim=1)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x9, batch)
        x = self.linear(x)        
        x = F.softmax(x, dim=1)

        return x



