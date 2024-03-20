from config import *

import os
from sklearn.feature_selection import SelectKBest, f_classif


def encode_fselector_filename(encode_method, impute_method, fs_method, sel_n_features):
    checkpoint_name = f'{fs_method}{sel_n_features}.pkl'
    parent_dir = os.path.join(
        FSELECTORS_DIR, encode_method, impute_method)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    filename = os.path.join(parent_dir, checkpoint_name)
    return filename


def feature_select_impl(X, y, fs_method, sel_n_features):
    if fs_method == "ftest":
        selector = SelectKBest(score_func=f_classif, k=sel_n_features)
        selector.fit(X, y)
        # boolean array of shape [# input features]
        support = selector.get_support()
    else:
        raise NotImplementedError("Unsupported feature selection method")

    return support
