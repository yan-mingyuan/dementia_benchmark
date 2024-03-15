from config import *

import numpy as np
import pandas as pd
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


def get_data_wave(wave):
    file_path = os.path.join(DIR_RW, f'w{wave}_subsets.dta')
    wave_data = pd.read_stata(file_path)

    # Columns to drop
    # cols_related = []
    cols_related = [
        'walkra', 'dressa', 'batha', 'eata', 'beda', 'adlwa',
        'phonea', 'moneya', 'medsa', 'shopa', 'mealsa', 'iadla',
        'tr20',
    ]
    cols_to_drop = SELF_DEM_COLS + PROXY_DEM_COLS + cols_related
    wave_data.drop(cols_to_drop, axis=1, inplace=True)

    features, labels = wave_data.drop(['demcls'], axis=1), wave_data['demcls']
    return features, labels


def encode(features, method="float"):
    encoded_features = features.copy()
    if method == "float":
        # Convert categorical variables to float
        for col in encoded_features.select_dtypes(include=['category']):
            encoded_features[col] = encoded_features[col].cat.codes.astype(
                float)
            encoded_features.loc[encoded_features[col]
                                 == -1, col] = float('nan')
    elif method == "dummy":
        # One-hot encoding
        encoded_features = pd.get_dummies(encoded_features, dummy_na=True)
        encoded_features = encoded_features.astype(
            float)  # Convert boolean columns to float
    else:
        raise NotImplementedError("Unsupported encoding method")

    return encoded_features


def impute(features, method="mode"):
    if method == "mode":
        imputed_features = features.fillna(
            features.mode().iloc[0], inplace=False)
    else:
        raise NotImplementedError("Unsupported imputation method")

    return imputed_features
