from config import *
from utlis import *

import itertools
import numpy as np
from sklearn.model_selection import train_test_split, KFold


class ClassificationDataset:
    def __init__(self, internal, external, encode_method, impute_method, random_state=42, test_ratio=0.2, n_split=5) -> None:
        # split data into multiple folds
        self.random_state = random_state

        # load data from file
        self.internal = internal
        self.external = external
        self.raw_features_dict, self.labels_dict = self.get_data_waves()

        # encode and impute
        self.encode_method = encode_method
        self.impute_method = impute_method
        self.imputed_features_dict = self.encode_and_impute()

        # normalize
        self.norm_features_dict = self.normalize()

        # split data
        self.test_ratio = test_ratio
        self.split = n_split
        self.data_train, self.data_test_internal, self.data_test_external = self.data_test_split()

    def get_data_waves(self):
        features_dict = {}
        labels_dict = {}
        for wave in [self.internal, self.external]:
            features, labels = get_data_wave(wave)

            features_dict[wave] = features
            labels_dict[wave] = labels
        return features_dict, labels_dict

    def encode_and_impute(self):
        features_dict = {}
        for wave in [self.internal, self.external]:
            raw_features = self.raw_features_dict[wave]
            # Encode for more imputation methods (e.g. median imputation)
            imputed_features = encode(raw_features, self.encode_method)
            imputed_features = impute(imputed_features, self.impute_method)

            features_dict[wave] = imputed_features
        return features_dict

    def normalize(self):
        features_dict = {}
        for wave in [self.internal, self.external]:
            features = self.imputed_features_dict[wave]
            mu, std = features.mean(), features.std()
            features_dict[wave] = (features - mu) / (std + 1e-8)
            min_val, max_val = features.min(), features.max()
            features_dict[wave] = (features - min_val) / (max_val - min_val + 1e-8)
        return features_dict

    def data_test_split(self):
        # Extract external test dataset
        features_external = self.norm_features_dict[self.external]
        labels_external = self.labels_dict[self.external]

        # Extract internal test dataset
        features = self.norm_features_dict[self.internal]
        labels = self.labels_dict[self.internal]
        features_tarin, features_internal, labels_train, labels_internal = train_test_split(
            features, labels, test_size=self.test_ratio, random_state=self.random_state)

        return (features_tarin, labels_train), (features_internal, labels_internal), (features_external, labels_external)

    def fold_generator(self):
        kf = KFold(n_splits=self.split, shuffle=True,
                   random_state=self.random_state)
        features, labels = self.data_train

        for train_index, val_index in kf.split(features, labels):
            train_features_split, train_labels_split = features.iloc[
                train_index], labels.iloc[train_index]
            val_features_split, val_labels_split = features.iloc[val_index], labels.iloc[val_index]
            yield (train_features_split, train_labels_split, val_features_split, val_labels_split)

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
        features_internal, labels_internal = self.data_test_internal
        features_external, labels_external = self.data_test_external

        # Evaluate "internal" test set
        preds_internal = best_model.predict(features_internal)
        metrics_internal = calculate_metrics(labels_internal, preds_internal)
        print_metrics(*metrics_internal, valid=False, internal=True)

        # Evaluate "external" test set
        preds_external = best_model.predict(features_external)
        metrics_external = calculate_metrics(labels_external, preds_external)
        print_metrics(*metrics_external, valid=False, internal=False)

        return metrics_internal, metrics_external
