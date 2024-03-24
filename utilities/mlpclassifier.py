import torch
import torch.nn as nn
import torch.optim as optim


class MLPClassifier(nn.Module):
    def __init__(
            self, hidden_layer_sizes=(100,), max_iter=200,
            weight_decay=1e-4, learning_rate=1e-3, momentum=0.9, nesterovs_momentum=True,
            batch_size=200, shuffle=True, random_state=42, cuda=True) -> None:
        super(MLPClassifier, self).__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.cuda_ = cuda

        self.model = None
        self.input_size = None
        self.output_size = 1

    def _build_model(self):
        layers = []
        layer_sizes = [self.input_size] + \
            list(self.hidden_layer_sizes) + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        if self.cuda_:
            self.model = self.model.cuda()

    def fit(self, X, y):
        self.input_size = X.shape[1]
        self._build_model()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if self.cuda_:
            X_tensor = X_tensor.cuda()
            y_tensor = y_tensor.cuda()

        self.criterion = nn.BCELoss(reduction='mean')
        self.optimizer = optim.SGD(
            self.model.parameters(), weight_decay=self.weight_decay, 
            lr=self.learning_rate, momentum=self.momentum, nesterov=self.nesterovs_momentum)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        self.model.train()
        for epoch in range(self.max_iter):
            for batch_idx, (inputs, targets) in enumerate(loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(1)
                loss = self.criterion(outputs, targets.float())
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)

        if self.cuda_:
            X_tensor = X_tensor.cuda()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        outputs = outputs.cpu().numpy()
        probs = np.concatenate([1 - outputs, outputs], 1)
        return probs


if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    def calculate_metrics(labels, probs):
        auc = roc_auc_score(labels, probs)
        aupr = average_precision_score(labels, probs)
        preds_categorical = np.where(
            probs > 0.5, 1.0, 0.0)
        acc = np.mean(preds_categorical == labels)
        return np.array([auc, aupr, acc])
    
    def print_metrics(auc, aupr, acc, valid=False, internal=True):
        if valid:
            print(f"Valid:         ", end='')
        else:
            if internal:
                print(f"Internal test: ", end='')
            else:
                print(f"External test: ", end='')
        print(
            f"AUC: {auc:.4f} | AUPR: {aupr:.4f} | Acc: {acc * 100:.2f}%")

    n_samples, n_features = 32, 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples).astype(np.float64)
    mlp_classifier = MLPClassifier(max_iter=1, cuda=False)
    mlp_classifier.fit(X, y)
    probs = mlp_classifier.predict_proba(X)[:, 1]
    metrics = calculate_metrics(y, probs)
    print_metrics(*metrics)
