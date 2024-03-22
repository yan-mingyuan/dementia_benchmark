from config import *

import re
import functools
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve


def get_model_name(model_fn):
    if isinstance(model_fn, functools.partial):
        model_fn = model_fn.func
    model_name = re.match(
        "<class '([^']+)'>", repr(model_fn)).group(1).split('.')[-1]
    return model_name


def encode_predictor_filename(encode_method, impute_method, fs_method, sel_n_features, model_name):
    checkpoint_name = f'{model_name}.pkl'
    parent_dir = os.path.join(
        PREDICTORS_DIR, encode_method, impute_method, f'{fs_method}{sel_n_features}')
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    filename = os.path.join(parent_dir, checkpoint_name)
    return filename


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
