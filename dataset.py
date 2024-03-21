from config import *
from utilities import (
    get_data_wave, encode_impl, Imputer, encode_imputer_filename, load_or_create_imputer,
    Normalizer, normalize_impl, encode_fselector_filename, feature_select_impl,
    calculate_metrics, print_metrics)

import gc
import pickle
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


class ClassificationDataset:
    def __init__(self, internal, external, encode_method, impute_method, fs_method, fs_ratio, norm_method, random_state=42, test_ratio=0.2, n_split=5, cached=True) -> None:
        self.random_state = random_state
        self.test_ratio = test_ratio
        self.split = n_split
        self.cached = cached

        self.internal = internal
        self.external = external
        self.encode_method = encode_method
        self.impute_method = impute_method
        self.fs_method = fs_method
        self.norm_method = norm_method

        # feature selection
        assert 0 < fs_ratio <= 1.0, "fs_ratio should be between 0 and 1."

        self.tot_data_dict, self.tot_n_features = self.load_and_process_data()
        self.sel_n_features = int(self.tot_n_features * fs_ratio)
        self.fs_data_dict, self.tot_columns, self.sel_columns = self.feature_select()

        gc.collect()

    def load_and_process_data(self):
        # load data from file
        raw_X_dict, self.y_dict = self.get_data_waves()

        # split data into multiple folds
        split_data_dict = self.data_test_split(raw_X_dict)

        # encode
        encoded_data_dict = self.encode(split_data_dict)
        # impute
        imputed_data_dict = self.impute(encoded_data_dict)
        # remove constant columns
        removed_data_dict, tot_n_features = self.remove_constant_columns(
            imputed_data_dict)

        # normalize
        norm_data_dict = self.normalize(removed_data_dict)

        return norm_data_dict, tot_n_features

    def get_data_waves(self):
        raw_X_dict = {}
        y_dict = {}
        for wave in [self.internal, self.external]:
            X, y = get_data_wave(wave)

            raw_X_dict[wave] = X
            y_dict[wave] = y
        return raw_X_dict, y_dict

    def data_test_split(self, raw_X_dict):
        splited_data_dict = {}

        # Extract external test dataset
        X_extest = raw_X_dict[self.external]
        y_extest = self.y_dict[self.external]
        splited_data_dict['extest'] = X_extest, y_extest

        # Extract internal test dataset
        X = raw_X_dict[self.internal]
        y = self.y_dict[self.internal]
        X_train, X_intest, y_train, y_intest = train_test_split(
            X, y, test_size=self.test_ratio, random_state=self.random_state)
        splited_data_dict['intest'] = X_intest, y_intest
        splited_data_dict['train'] = X_train, y_train

        return splited_data_dict

    def encode(self, split_data_dict):
        encoded_data_dict = {}
        for split, (X, y) in split_data_dict.items():
            # Encode for more imputation methods (e.g. median imputation)
            X_encoded = encode_impl(X, self.encode_method)
            encoded_data_dict[split] = (X_encoded, y)

        return encoded_data_dict

    def impute(self, encoded_data_dict):
        X_train, y_train = encoded_data_dict['train']

        filename = encode_imputer_filename(
            self.encode_method, self.impute_method) if self.cached else None
        imputer, data_cached = load_or_create_imputer(
            X_train, y_train, self.impute_method, filename)

        if data_cached:
            # Load cached data for knn imputation method to avoid slow inference
            with open(filename, 'rb') as fp:
                imputed_data_dict = pickle.load(fp)
        else:
            imputer: Imputer
            imputed_data_dict = {}
            for split, (X, y) in encoded_data_dict.items():
                X_imputed = imputer.transform(X)
                X_imputed = pd.DataFrame(X_imputed, columns=X_train.columns)
                imputed_data_dict[split] = (X_imputed, y)
            if data_cached:
                with open(filename, 'wb') as fp:
                    pickle.dump(imputed_data_dict, fp)

        return imputed_data_dict

    def remove_constant_columns(self, imputed_data_dict):
        X_train = imputed_data_dict['train'][0]

        constant_columns = []
        for column in X_train.columns:
            if X_train[column].nunique() == 1:
                constant_columns.append(column)

        removed_data_dict = {}
        for split, (X, y) in imputed_data_dict.items():
            X_removed = X.drop(
                columns=constant_columns, inplace=False)
            removed_data_dict[split] = (X_removed, y)
        tot_n_features = X_removed.shape[1]

        return removed_data_dict, tot_n_features

    def normalize(self, removed_data_dict):
        X_train = removed_data_dict['train'][0]
        normalizer: Normalizer = normalize_impl(X_train, self.norm_method)

        norm_data_dict = {}
        for split, (X, y) in removed_data_dict.items():
            X_norm = normalizer.transform(X)
            norm_data_dict[split] = (X_norm, y)

        return norm_data_dict

    def feature_select(self):
        X_train, y_train = self.tot_data_dict['train']
        tot_columns = X_train.columns

        filename = encode_fselector_filename(
            self.encode_method, self.impute_method, self.fs_method, self.sel_n_features) if self.cached else None
        if filename and os.path.exists(filename):
            with open(filename, 'rb') as f:
                selector = pickle.load(f)
        else:
            selector = feature_select_impl(
                X_train.values, y_train.values, self.fs_method, self.sel_n_features)
            if self.cached:
                with open(filename, 'wb') as f:
                    pickle.dump(selector, f)

        # boolean array of shape [# input features]
        support: np.ndarray[bool] = selector.get_support()
        # Select columns based on feature selection support
        sel_columns = tot_columns[support]

        fs_data_dict = {}
        for split, (X, y) in self.tot_data_dict.items():
            X_fs = X[sel_columns]
            # Convert DataFrame to ndarray
            fs_data_dict[split] = (X_fs.values, y.values)

        return fs_data_dict, tot_columns, sel_columns

    def fold_generator(self):
        kf = KFold(n_splits=self.split, shuffle=True,
                   random_state=self.random_state)
        X, y = self.fs_data_dict['train']

        for train_index, val_index in kf.split(X, y):
            X_train, y_train = X[train_index], y[train_index]
            X_val, y_val = X[val_index], y[val_index]
            yield (X_train, y_train, X_val, y_val)

    def perform_grid_search(self, model_fn, param_grid_list, verbose=True):
        best_metrics = [-np.inf] * 3
        best_params = None
        best_model = None
        for param_grid in param_grid_list:
            keys = param_grid.keys()
            for vals in itertools.product(*param_grid.values()):
                params = dict(zip(keys, vals))
                metrics_lst = []
                for X_train, y_train, X_val, y_val in self.fold_generator():
                    # Train
                    model = model_fn(**params)
                    model.fit(X_train, y_train)

                    # Evaluate
                    preds = model.predict(X_val)
                    metrics = calculate_metrics(y_val, preds)
                    metrics_lst.append(metrics)

                # Calculate average score
                metrics_mean = np.mean(metrics_lst, 0)

                if verbose:
                    print(
                        f"model({', '.join([f'{param_name}={param_value}' for param_name, param_value in params.items()])})")
                    print_metrics(*metrics_mean, valid=True)

                # Update best score and parameters if necessary
                if metrics_mean[0] > best_metrics[0]:
                    best_metrics = metrics_mean
                    best_params = params
                    best_model = model

        print("=======================================================")
        print(
            f"best model({', '.join([f'{param_name}={param_value}' for param_name, param_value in best_params.items()])})")
        print_metrics(*best_metrics, valid=True)
        return best_metrics, best_params, best_model

    def evaluate_test_sets(self, best_model):
        # Get internal and external test sets
        X_intest, y_intest = self.fs_data_dict['intest']
        X_extest, y_extest = self.fs_data_dict['extest']

        # Evaluate "internal" test set
        y_hat_intest = best_model.predict(X_intest)
        metrics_intest = calculate_metrics(y_intest, y_hat_intest)
        print_metrics(*metrics_intest, valid=False, internal=True)

        # Evaluate "external" test set
        y_hat_extest = best_model.predict(X_extest)
        metrics_extest = calculate_metrics(y_extest, y_hat_extest)
        print_metrics(*metrics_extest, valid=False, internal=False)

        return metrics_intest, metrics_extest

    def __repr__(self):
        fs_features_train = len(self.fs_data_dict['train'][0])
        fs_features_intest = len(self.fs_data_dict['intest'][0])
        fs_features_extest = len(self.fs_data_dict['extest'][0])

        return (
            f"ClassificationDataset(\n"
            f"    Train: {fs_features_train},\n"
            f"    Internal Test: {fs_features_intest},\n"
            f"    External Test: {fs_features_extest}\n"
            f"    Selected Features: {self.sel_n_features}/{self.tot_n_features}\n"
            "  )"
        )


if __name__ == "__main__":
    internal, external = 11, 12
    encode_method = "dummy"
    impute_method = "mice"
    fs_method, fs_ratio = "ftest", 0.5
    norm_method = "zscore"
    classification_dataset = ClassificationDataset(
        internal, external, encode_method, impute_method, fs_method, fs_ratio, norm_method, random_state=SEED)
    print(classification_dataset)
