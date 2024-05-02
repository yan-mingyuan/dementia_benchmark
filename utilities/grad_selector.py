from .base_transformer import FeatureSelector
from .nnblocks import MLP, GNN

import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


class GradSelector(nn.Module):
    def __init__(self, backbone):
        super(GradSelector, self).__init__()
        self.backbone = backbone

        # deactivate input layer's bias
        self.backbone.input.bias = None

        # multiple dropout
        self.backbone.multiple_dropout = True

    def forward(self, x):
        return self.backbone(x)


class SelectFromGrad(FeatureSelector):
    def __init__(self, fs_method, max_features, n_dropout=20, dropout_p=0.50, alpha=0.95,
                 hidden_layer_sizes=None, max_iter=5,
                 weight_decay=1e-4, learning_rate=5e-3, momentum=0.0, squares=0.999, optimizer_t='sgdm',
                 batch_size=512, shuffle=True, random_state=42, cuda=True) -> None:
        super().__init__()

        # Selector configuration
        assert fs_method in ['dnp', 'graces']
        self.fs_method = fs_method
        self.max_features = max_features
        self.n_dropout = n_dropout
        self.alpha = alpha
        self.q_norm = 2
        self.xavier_std = None
        self.support_indices = [0]
        self.support = None

        # Model configuration
        self.dropout_p = dropout_p
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64, 64]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.input_size = None
        self.output_size = 1
        self.model = None
        self.criterion = nn.BCELoss(reduction='mean')

        # Optimizier configuration
        # Cannot too big, otherwise gradients of all features are zeros
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        # Traditional momentum maybe not suitable for this task
        # NOTE: Any momentum greater than zero can be buggy in this situation
        self.momentum = momentum
        self.squares = squares
        self.optimizer_t = optimizer_t

        # Environment configuration
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.cuda_ = cuda

    def _build_model(self):
        self.support = np.zeros(self.input_size-1, dtype=bool)
        self.xavier_std = math.sqrt(
            4. / self.input_size + self.hidden_layer_sizes[0])

        match self.fs_method:
            case 'dnp':
                backbone = MLP(self.input_size, self.hidden_layer_sizes,
                               self.output_size, self.dropout_p)
            case 'graces':
                backbone = GNN(self.input_size, self.hidden_layer_sizes,
                               self.output_size, self.dropout_p, self.alpha)
            case _: raise NotImplementedError
        self.model = GradSelector(backbone)

        if self.cuda_:
            self.model = self.model.cuda()

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        # Add intercept
        X_tensor = torch.cat([torch.ones(X_tensor.size(0), 1), X_tensor], 1)

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

        for _ in tqdm(range(self.max_features), desc="Number of Selected Features", leave=True):
            # for _ in range(self.max_features):
            self.model.train()
            for epoch in range(self.max_iter):
                # with tqdm(loader, desc=f"Epoch {epoch+1}/{self.max_iter}", unit="batch", leave=False) as tepoch:
                #     for batch_idx, (inputs, targets) in enumerate(tepoch):
                for batch_idx, (inputs, targets) in enumerate(loader):
                    # if self.cuda_:
                    #     inputs = inputs.cuda()
                    #     targets = targets.cuda()

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets.float())
                    loss.backward()
                    self.optimizer.step()

                # NOTE: Set value of unselected input weights to be zeros
                with torch.no_grad():
                    for s in range(self.input_size):
                        if s not in self.support_indices:
                            self.model.backbone.input.weight[:, s].zero_()

            self.model.eval()
            for _ in range(self.n_dropout):
                for batch_idx, (inputs, targets) in enumerate(loader):
                    # if self.cuda_:
                    #     inputs = inputs.cuda()
                    #     targets = targets.cuda()

                    # NOTE: loss is accumulated but not update gradients
                    # self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets.float())
                    loss.backward()
                    # self.optimizer.step()
            # input_gradient = self.model.backbone.input.weight.grad.detach() / self.n_dropout
            input_gradient = self.model.backbone.input.weight.grad.detach()
            self.optimizer.zero_grad()
            gradient_norm = input_gradient.norm(p=self.q_norm, dim=0)
            # gradient_norm[self.support_indices] = 0.0
            gradient_norm[self.support_indices] = -np.inf
            new_s = torch.argmax(gradient_norm).item()
            self.support_indices.append(new_s)

            with torch.no_grad():

                # for s in range(self.input_size):
                #     if s == new_s:
                #         # Xavier normalization
                #         self.model.backbone.input.weight[:, s].normal_(
                #             0, self.xavier_std)
                #     elif s not in self.S:
                #         # Remains zero
                #         # TODO: not necessary because gradients have been cleared after the last epoch
                #         # self.model.backbone.input.weight[:, s].zero_()
                #         pass

                # Xavier normalization
                self.model.backbone.input.weight[:, new_s].normal_(0, self.xavier_std)
        # print(self.support_indices)

        # remove intercept
        for s in self.support_indices[1:]:
            self.support[s-1] = 1

    def get_support(self):
        return self.support


if __name__ == "__main__":
    q = 2
    n_dropout = 5
    criterion = nn.CrossEntropyLoss()

    bsz, in_c, dim, n_classes = 16, 3, 8, 10
    xavier_std = math.sqrt(4. / in_c + dim)

    x = torch.randn(bsz, in_c)
    y = torch.randint(n_classes, (bsz, ))
    net = nn.Sequential(
        nn.Linear(in_c, dim, bias=False),
        nn.Dropout(.5),
        nn.ReLU(),
        nn.Linear(dim, n_classes),
    )

    S = [0]

    # fix W_c
    # with torch.no_grad():
    #     for s in range(in_c):
    #         if s not in S:
    #             net[0].weight[:, s].requires_grad_(False)

    net.eval()
    for _ in range(n_dropout):
        loss = criterion(net(x), y)
        loss.backward()
        # print(net[0].weight.grad[:, 1])
        # print(net[0].weight.grad.mean())
    net.zero_grad()

    for _ in range(n_dropout):
        loss = criterion(net(x), y)
        loss.backward()
        # print(net[0].weight.grad[:, 1])
        # print(net[0].weight.grad.mean())

    input_gradient = net[0].weight.grad.detach() / n_dropout
    gradient_norm = input_gradient.norm(p=q, dim=0)
    gradient_norm[S] = 0

    new_s = torch.argmax(gradient_norm).item()
    S.append(new_s)

    with torch.no_grad():
        for i in range(in_c):
            if i == new_s:
                # Xavier normalization
                net[0].weight[:, i].normal_(0, xavier_std)
            elif i not in S:
                # remain zero
                net[0].weight[:, i].zero_()

    # print(net[0].weight)
