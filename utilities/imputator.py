from .data_transformer import DataTransformer

import pandas as pd
# from statsmodels.imputation.mice import MICE
from fancyimpute import IterativeImputer


class Imputer(DataTransformer):
    pass


class ModeImputer(Imputer):
    def fit(self, X, y=None):
        self.modes = X.mode().iloc[0]

    def transform(self, X: pd.DataFrame):
        dataframe = X.fillna(self.modes, inplace=False)
        values = dataframe.values
        return values


def impute_impl(X, y=None, method="mode"):
    if method == "mode":
        imputer = ModeImputer(X)
    elif method == "mice":
        imputer = IterativeImputer(max_iter=0)
        imputer.fit(X)
        # imputed_features = pd.DataFrame(
        #     imputer.transform(X), columns=X.columns)
        # print(imputed_features)
        # print(X.iloc[11:16, 10:16])
        # print(imputed_features.iloc[11:16, 10:16])
        # imputed_features = mice.complete(features.values)
        # imputed_features = mice.fit_transform(features)
    else:
        raise NotImplementedError("Unsupported imputation method")

    return imputer
