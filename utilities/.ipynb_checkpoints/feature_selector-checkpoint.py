from config import *
from .base_transformer import FeatureSelector
from .grad_selector import SelectFromGrad
from .cancelout_selector import SelectFromOut

import os
from functools import partial
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def encode_fselector_filename(encode_method, impute_method, fs_method, sel_n_features):
    checkpoint_name = f'{fs_method}{sel_n_features}.pkl'
    parent_dir = os.path.join(
        FSELECTORS_DIR, encode_method, impute_method)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    filename = os.path.join(parent_dir, checkpoint_name)
    return filename


def feature_select_impl(X, y, fs_method, sel_n_features) -> FeatureSelector:
    if fs_method in ['fcls', 'freg', 'chi2', 'micls', 'mireg']:
        score_func_dict = {
            'fcls': f_classif,
            # 'f1w': f_oneway,
            # 'freg': f_regression, # Same as ANOVA when y is binary variable
            # This method doesn't use absolute values, so results might be incorrect
            # 'rreg': r_regression
            'chi2': chi2,
            'micls': partial(mutual_info_classif, n_neighbors=3, random_state=SEED),
            'mireg': partial(mutual_info_regression, n_neighbors=3, random_state=SEED),
        }
        score_func = score_func_dict.get(fs_method)
        selector = SelectKBest(score_func=score_func, k=sel_n_features)
    elif fs_method in ['lasso', 'svc', 'rfcls']:
        estimator_dict = {
            # 'lasso': Lasso(random_state=SEED, alpha=0.0),
            'lasso': LogisticRegression(random_state=SEED, n_jobs=1, class_weight='balanced', solver='liblinear', penalty='l1', C=1.0, max_iter=2000),
            # kernel method is not allowed
            'svc': LinearSVC(random_state=SEED, dual="auto", class_weight='balanced', C=1.0),
            'rfcls': RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight='balanced', n_estimators=100),
            # 'rfreg': RandomForestRegressor(random_state=SEED, n_jobs=-1, n_estimators=100),
        }
        estimator = estimator_dict.get(fs_method)
        selector = SelectFromModel(estimator, max_features=sel_n_features)
    elif fs_method in ['dnp', 'graces', 'cancelout', 'graphout']:
        if fs_method in ['dnp', 'graces']:
            selector = SelectFromGrad(fs_method, max_features=sel_n_features)
        elif fs_method in ['cancelout', 'graphout']:
            selector = SelectFromOut(fs_method, max_features=sel_n_features)
    else:
        raise NotImplementedError("Unsupported feature selection method")

    selector.fit(X, y)
    return selector
