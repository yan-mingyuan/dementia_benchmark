import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class _BasicBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=True, bias=True):
        super(_BasicBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size, bias),
        )

    def forward(self, x):
        out = self.block(self.norm(x))
        return out
    
class _GraphBasicBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=True, bias=True):
        super(_GraphBasicBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()

        self.block = gnn.Sequential(
            (gnn.SAGEConv(input_size, output_size), 'x, edge_index -> x'),
            # gnn.SAGEConv(input_size, output_size, bias),
        )

    def forward(self, x, edge_index):
        out = self.block(self.norm(x), edge_index)
        return out


class _ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=True, bias=True):
        super(_ResidualBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(output_size, output_size, bias=bias),
        )

        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, output_size, bias),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Identity(),
            )

    def forward(self, x):
        residual = x
        out = self.block(self.norm(x))
        out += self.shortcut(residual)
        return out

class _GraphResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=True, bias=True):
        super(_GraphResidualBlock, self).__init__()

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()
        
        self.block = gnn.Sequential('x, edge_index', [
            (gnn.SAGEConv(input_size, output_size), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (gnn.SAGEConv(output_size, output_size), 'x, edge_index -> x'),
        ])

        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, output_size, bias),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Identity(),
            )

    def forward(self, x, edge_index):
        residual = x
        out = self.block(self.norm(x), edge_index)
        out += self.shortcut(residual)
        return out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, use_residual=True, use_batch_norm=False):
        super(MLP, self).__init__()

        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        if self.use_residual:
            self.block = _ResidualBlock
        else:
            self.block = _BasicBlock
        self.block = functools.partial(
            self.block, use_batch_norm=self.use_batch_norm)

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.n_hidden = len(self.hidden_layer_sizes) - 1

        # Set bias to False
        # self.input.bias = None
        self.input = self.block(
            self.input_size, self.hidden_layer_sizes[0], bias=True)
        self.hiddens = nn.ModuleList([
            self.block(hidden_layer_sizes[h], hidden_layer_sizes[h+1])
            for h in range(self.n_hidden)
        ])
        self.output = self.block(hidden_layer_sizes[-1], output_size)

    def forward(self, x):
        x = F.relu(self.input(x), inplace=True)
        for hidden in self.hiddens:
            x = F.relu(hidden(x), inplace=True)
        x = F.sigmoid(self.output(x))
        return x.squeeze(1)


class MLPSelector(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_p=0.25, use_residual=True, use_batch_norm=False):
        super(MLPSelector, self).__init__()

        # Batchnorm and Residual may be over-complicated for DNP
        # Never mind if we have a good implementation
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        if self.use_residual:
            self.block = _ResidualBlock
        else:
            self.block = _BasicBlock
        self.block = functools.partial(
            self.block, use_batch_norm=self.use_batch_norm)

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.n_hidden = len(self.hidden_layer_sizes) - 1

        # NOTE: In input layer, bias is False and use Linear only
        # TODO: move the input layer out of MLPSelector
        self.input = nn.Linear(
            self.input_size, self.hidden_layer_sizes[0], bias=False)
        self.hiddens = nn.ModuleList([
            self.block(hidden_layer_sizes[h], hidden_layer_sizes[h+1])
            for h in range(self.n_hidden)
        ])
        self.output = self.block(hidden_layer_sizes[-1], output_size)

        # Dropout
        #     1. drop neurons rather than weights;
        #     2. apply dropout in eval stage;
        #     3. must dropout for each layer?
        self.dropout_p = dropout_p
        self.dropout = functools.partial(
            F.dropout, p=self.dropout_p, training=not self.training)

    def forward(self, x):
        x = F.relu(self.input(x), inplace=True)
        for hidden in self.hiddens:
            x = F.relu(hidden(x), inplace=True)
            x = self.dropout(x)
        x = F.sigmoid(self.output(x))
        return x.squeeze(1)


class GraphSelector(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_p=0.25, alpha=0.95, use_residual=True, use_batch_norm=True):
        super(GraphSelector, self).__init__()

        # Batchnorm and Residual may be over-complicated for DNP
        # Never mind if we have a good implementation
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        if self.use_residual:
            self.block = _GraphResidualBlock
        else:
            self.block = _GraphBasicBlock
        self.block = functools.partial(
            self.block, use_batch_norm=self.use_batch_norm)

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.n_hidden = len(self.hidden_layer_sizes) - 1

        # NOTE: In input layer, bias is False
        # TODO: move the input layer out of MLPSelector
        self.input = nn.Linear(
            self.input_size, self.hidden_layer_sizes[0], bias=False)
        self.hiddens = nn.ModuleList([
            self.block(hidden_layer_sizes[h], hidden_layer_sizes[h+1])
            for h in range(self.n_hidden)
        ])
        self.output = self.block(hidden_layer_sizes[-1], output_size)

        # Dropout
        #     1. drop neurons rather than weights;
        #     2. apply dropout in eval stage;
        #     3. must dropout for each layer?
        self.dropout_p = dropout_p
        self.dropout = functools.partial(
            F.dropout, p=self.dropout_p, training=not self.training)
        self.alpha = alpha

    def forward(self, x):
        edge_index = self.create_edge_index(x)

        x = F.relu(self.input(x), inplace=True)
        for hidden in self.hiddens:
            x = F.relu(hidden(x, edge_index), inplace=True)
            x = self.dropout(x)
        x = F.sigmoid(self.output(x, edge_index))
        return x.squeeze(1)

    def create_edge_index(self, x):
        similarity_matrix = torch.abs(F.cosine_similarity(
            x[None, :, :], x[:, None, :], dim=-1))
        # similarity_matrix = torch.abs(F.cosine_similarity(
        #     x[..., None, :, :], x[..., :, None, :], dim=-1))
        similarity = torch.sort(similarity_matrix.view(-1))[0]
        eps = torch.quantile(similarity, self.alpha, interpolation='nearest')
        adj_matrix = similarity_matrix >= eps
        row, col = torch.where(adj_matrix)
        edge_index = torch.cat((row.reshape(1, -1), col.reshape(1, -1)), dim=0)
        return edge_index
