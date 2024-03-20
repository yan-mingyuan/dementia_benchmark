from .data_transformer import DataTransformer

from config import *

import re
import pickle
# import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


class Imputer(DataTransformer):
    pass


def encode_imputer_filename(encode_method, impute_method):
    parent_dir = os.path.join(IMPUTERS_DIR, encode_method)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    checkpoint_name = impute_method + '.pkl'
    filename = os.path.join(parent_dir, checkpoint_name)
    return filename


def load_or_create_imputer(X, y=None, method="mode", filename=None):
    data_cached = False
    if method in ['mean', 'median', 'mode']:
        imputer = _train_simple_imputer(X, imputer)
    elif method.startswith('knn'):
        # Cache for knn-imputed data, as it is time-consuming to inference
        imputer, data_cached = _train_knn_imputer(X, method, filename)
    elif method.startswith("it"):
        # Cache for iterative model, as it is time-consuming to train
        imputer = _train_iterative_imputer(X, method, filename)
    else:
        raise NotImplementedError("Unsupported imputation method")

    return imputer, data_cached


def _train_simple_imputer(X, method) -> Imputer:
    strategy = 'most_frequent' if method == "mode" else method
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X)
    return imputer


def _train_knn_imputer(X, method, filename=None):
    imputer = None
    # Check if a cached imputer exists, if so, load it and return
    if filename and os.path.exists(filename):
        data_cached = True
    else:
        data_cached = False

        pattern = r'(knn)(\d+)([ud])'
        matches = re.match(pattern, method)

        n_neighbors = int(matches.group(2))
        weights = "distance" if matches.group(3) == 'd' else "uniform"
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputer.fit(X)

    return imputer, data_cached


def _train_iterative_imputer(X, method, filename=None) -> Imputer:
    # Check if a cached imputer exists, if so, load it and return
    if filename and os.path.exists(filename):
        with open(filename, 'rb') as fp:
            imputer = pickle.load(fp)
        return imputer

    # Parse the method string to extract model and max_iter information
    pattern = r'(it)([a-zA-Z]+)(\d*)'
    matches = re.match(pattern, method)
    model = matches.group(2)
    max_iter = int(matches.group(3)) if matches.group(3) else 10

    # Define dictionary mapping model names to corresponding estimator objects
    estimator_dict = {
        "lr": LinearRegression(n_jobs=-1),
        "ridge": Ridge(random_state=SEED, alpha=1.0),
        "lasso": Lasso(random_state=SEED, alpha=1.0),
        "enet": ElasticNet(random_state=SEED, alpha=1.0, l1_ratio=0.5, max_iter=2000, selection="random"),
        "br": BayesianRidge(),  # default
        'ard': ARDRegression(),
        "dt": DecisionTreeRegressor(random_state=SEED),
        "svm": SVR(kernel='rbf'),
        "lsvm": LinearSVR(random_state=SEED, dual='auto', max_iter=1000),
        # Kernel crashed while executing code
        # "rf": RandomForestRegressor(random_state=SEED, n_jobs=-1, n_estimators=30),
        # "ada": AdaBoostRegressor(DecisionTreeRegressor(random_state=SEED), random_state=SEED),
        "bagdt": BaggingRegressor(DecisionTreeRegressor(random_state=SEED), random_state=SEED, n_jobs=-1),
        # "gbm": GradientBoostingRegressor(random_state=SEED),
        "histgbm": HistGradientBoostingRegressor(random_state=SEED),
        "xgbm": XGBRegressor(random_state=SEED, n_jobs=-1),  # add weights
        "lgbm": LGBMRegressor(verbosity=-1, random_state=SEED, n_jobs=-1, class_weight="balanced"),
    }

    # Retrieve the estimator object based on the parsed model name
    estimator = estimator_dict.get(model)
    if estimator is None:
        raise NotImplementedError("Unsupported iterative imputer model")
    # Build the pipeline
    if model in ["br", "dt", "rf"]:
        # Some models don't require standard scaler
        pipeline = estimator
    else:
        # Other models require standard scaler
        pipeline = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('estimator', estimator)
        ])

    # Train the IterativeImputer with the constructed pipeline
    imputer = IterativeImputer(
        estimator=pipeline, max_iter=max_iter, initial_strategy="mean", random_state=SEED)
    imputer.fit(X)

    # Cache the trained imputer model if a filename is provided
    if filename:
        with open(filename, 'wb') as fp:
            pickle.dump(imputer, fp)
    return imputer
