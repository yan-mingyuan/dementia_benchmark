from abc import ABC, abstractmethod


class DataTransformer(ABC):
    def __init__(self, X, y=None) -> None:
        super().__init__()
        self.fit(X, y)

    @abstractmethod
    def fit(self, X, y=None, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, X):
        pass


class FeatureSelector(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def get_support(self):
        pass

# if __name__ == "__main__":
#     DataTransformer()
