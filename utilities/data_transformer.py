from abc import ABC, abstractmethod


class DataTransformer(ABC):
    def __init__(self, X, y=None) -> None:
        super().__init__()
        self.fit(X, y)

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass
