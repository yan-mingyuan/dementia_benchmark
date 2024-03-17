import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve


def calculate_metrics(labels, preds):
    auc = roc_auc_score(labels, preds)
    aupr = average_precision_score(labels, preds)
    preds_categorical = np.where(
        preds > 0.5, 1.0, 0.0)
    acc = np.mean(preds_categorical == labels)
    return auc, aupr, acc


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
