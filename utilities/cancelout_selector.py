from .base_transformer import FeatureSelector
from .nnblocks import MLP, GraphNet

import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


class CancelOut(nn.Module):
    def __init__(self, input_size, backbone):
        super(CancelOut, self).__init__()
        self.cancelout = nn.Parameter(
            torch.zeros(input_size, requires_grad=True) + 4)
        self.backbone = backbone

    def forward(self, x):
        x = x * torch.sigmoid(self.cancelout.float())
        x = self.backbone(x)
        return x


class SelectFromCancelOut(FeatureSelector):
    def __init__(self, fs_method, max_features, l1_penalty=1e-3, var_penalty=1e-3,
                 hidden_layer_sizes=None, max_iter=100,
                 weight_decay=1e-4, learning_rate=5e-3, momentum=0.9, squares=0.999, optimizer_t='sgdm',
                 use_residual=True, use_batch_norm=False,
                 batch_size=512, shuffle=True, random_state=42, cuda=True) -> None:
        super().__init__()

        # Selector configuration
        self.l1_penalty = l1_penalty
        self.var_penalty = var_penalty

        assert fs_method in ['cancelout', 'graphout']
        self.fs_method = fs_method
        self.max_features = max_features
        self.support_indices = None
        self.support = None

        # Model configuration
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64, 64]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.input_size = None
        self.output_size = 1
        if use_residual is None:
            use_residual = True
        self.use_residual = use_residual
        if use_batch_norm is None:
            use_batch_norm = (fs_method != 'cancelout')
        self.use_batch_norm = use_batch_norm
        self.model = None
        self.criterion = nn.BCELoss(reduction='mean')

        # Optimizier configuration
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.squares = squares
        self.optimizer_t = optimizer_t

        # Environment configuration
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.cuda_ = cuda

    def _build_model(self):
        self.support = np.zeros(self.input_size, dtype=bool)

        if self.fs_method == 'cancelout':
            backbone = MLP(
                self.input_size, self.hidden_layer_sizes, self.output_size)
        elif self.fs_method == 'graphout':
            backbone = GraphNet(
                self.input_size, self.hidden_layer_sizes, self.output_size)
        else:
            raise NotImplementedError
        self.model = CancelOut(self.input_size, backbone)

        if self.cuda_:
            self.model = self.model.cuda()

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        self.input_size = X_tensor.size(1)
        self._build_model()

        # TODO: remove it to avoid out of memory
        # Keep it when dataset is small to make learning run faster
        if self.cuda_:
            X_tensor = X_tensor.cuda()
            y_tensor = y_tensor.cuda()

        match self.optimizer_t:
            case 'sgdm':
                # NOTE: nesterov=False to avoid bad thing happens
                self.optimizer = optim.SGD(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                    momentum=self.momentum, nesterov=False)
            case 'adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                    betas=(self.momentum, self.squares))
            case _:
                raise NotImplementedError(
                    "Optimizer '{}' not implemented.".format(self.optimizer_t))

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        self.model.train()
        for epoch in range(self.max_iter):
            for batch_idx, (inputs, targets) in enumerate(loader):
                # if self.cuda_:
                #     inputs = inputs.cuda()
                #     targets = targets.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets.float())

                cancelout_weights_importance = self.model.cancelout
                l1_loss = self.l1_penalty * \
                    torch.norm(torch.sigmoid(cancelout_weights_importance), 1)
                var_loss = -self.var_penalty * \
                    torch.var(cancelout_weights_importance)
                loss += l1_loss + var_loss

                loss.backward()
                self.optimizer.step()

        self.support_indices = cancelout_weights_importance.detach().argsort(
            descending=True).cpu().numpy()[:self.max_features]
        for s in self.support_indices:
            self.support[s] = 1

    def get_support(self):
        return self.support
