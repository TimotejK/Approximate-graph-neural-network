import torch.nn as nn
import torch.nn.functional as F

# torch geometric libraries
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import optim
import torch_geometric.utils as pyg_utils

from implementation.custom_convolution import CustomConv, CustomLinear


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNStack, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        for l in range(2):
            self.lns.append(nn.LayerNorm(hidden_dim))
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            CustomLinear(hidden_dim, hidden_dim, bias=True), nn.Dropout(0.25),
            CustomLinear(hidden_dim, output_dim, bias=True))

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        return CustomConv(input_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)