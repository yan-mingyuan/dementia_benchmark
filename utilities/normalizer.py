from .data_transformer import DataTransformer


class Normalizer(DataTransformer):
    pass


class MaxMinNormalizer(Normalizer):
    def fit(self, X, y=None):
        self.min_val = X.min()
        self.range_ = X.max() - self.min_val + 1e-8

    def transform(self, X):
        return (X - self.min_val) / self.range_


class ZScoreNormalizer(Normalizer):
    def fit(self, X, y=None):
        self.mu = X.mean()
        self.std = X.std()

    def transform(self, X):
        return (X - self.mu) / (self.std + 1e-8)


def normalize_impl(X, norm_method):
    if norm_method == "maxmin":
        normalizer = MaxMinNormalizer(X)
    elif norm_method == "zscore":
        normalizer = ZScoreNormalizer(X)
    else:
        raise NotImplementedError("Unsupported normalization method")
    return normalizer
