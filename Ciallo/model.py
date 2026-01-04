import torch
import torch.nn.functional as F
from typing import Final, Union, Tuple
from copy import deepcopy
from attention_nn import GATv3Conv
from torch import nn as nn
from torch_geometric.nn import GatedGraphConv, Sequential
from torch_geometric.nn.models import GAT, GCN, GIN
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.aggr import MeanAggregation


class GraphConvClassifier(torch.nn.Module):
    def __init__(self, arg) -> None:
        super(GraphConvClassifier, self).__init__()
        self.arg = arg
        # self.embedding = torch.nn.Embedding(vocab_size, arg.embed_dim, padding_idx=pad_idx)
        if arg.conv_layer == "ggnn":
            self.graph = GatedGraphConv(
                out_channels=arg.embed_dim,
                num_layers=arg.num_layers,
            )
        elif arg.conv_layer == "gat":
            self.graph = GAT(
                in_channels=arg.embed_dim,
                hidden_channels=arg.embed_dim,
                out_channels=arg.embed_dim,
                num_layers=arg.num_layers,
                dropout=0.2,
                v2=True,
                heads=arg.head,
            )
        elif arg.conv_layer == "gcn":
            self.graph = GCN(
                in_channels=arg.embed_dim,
                hidden_channels=arg.embed_dim,
                out_channels=arg.embed_dim,
                num_layers=arg.num_layers,
                dropout=0.2,
            )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(arg.embed_dim, arg.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(arg.embed_dim, 1),
        )

    def forward(self, x, edge_index, batch, mask):
        x = self.graph(x, edge_index)
        x = self.linear(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        for gid in torch.unique(batch):
            nodes_in_target_graph = (batch == gid).nonzero(as_tuple=True)[0]
            # may need mask to control the meanless code.
            x[nodes_in_target_graph] = F.softmax(x[nodes_in_target_graph], dim=-1)
        return x


    def logits(self, x, edge_index, batch, mask):
        x = self.graph(x, edge_index)
        x = self.linear(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        return x        


    def encode(self, x, edge_index, batch, mask):
        x = self.graph(x, edge_index)
        return x
