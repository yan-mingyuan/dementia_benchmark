from config import *
from utilities import get_model_name
from dataset import ClassificationDataset

import functools
import numpy as np
import pandas as pd

results_dict = {}
internal, external = 11, 12
encode_method = "dummy"
impute_method = "itard"
fs_method, fs_ratio = "chi2", 0.5
norm_method = "maxmin"
classification_dataset = ClassificationDataset(
    internal, external, encode_method, impute_method, fs_method, fs_ratio, norm_method, random_state=SEED)

from utilities import GNNClassifier

model_fn = functools.partial(
    GNNClassifier, random_state=SEED, max_iter=50,
    momentum=0.9, squares=0.999, optimizer_t='sgdm',
    use_residual=True, use_batch_norm=False)
param_grid_list = [{
    'hidden_layer_sizes': [(128, 128), (64, 64, 64)],
    'learning_rate': [5e-3, 1e-2, 2e-2],
}]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model_fn, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

import warnings
from pytorch_tabnet.tab_model import TabNetClassifier
warnings.filterwarnings("ignore", category=UserWarning)

model_fn = functools.partial(
    TabNetClassifier, seed=SEED, 
    n_d=4, n_a=4, momentum=0.05, 
    n_steps=3, gamma=1.75, cat_emb_dim=1,
    n_independent=2, n_shared=2,
    verbose=0)
param_grid_list = [{
    'optimizer_params': [dict(lr=lr) for lr in [5e-2]],
    'gamma': [1.5, 1.75, 2.0],
}]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model_fn, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)
warnings.resetwarnings()

# from sklearn.neural_network import MLPClassifier
from utilities import MLPClassifier

model_fn = functools.partial(
    MLPClassifier, random_state=SEED, max_iter=50,
    momentum=0.9, squares=0.999, optimizer_t='sgdm',
    use_residual=True, use_batch_norm=False)
param_grid_list = [{
    # hidden_layer_sizes = (64, 32)
    # config       lr_range            worst_auc
    # adam w/ bn:  [2e-5, 5e-5, 1e-4]  0.8942
    # adam w/o bn: [5e-5, 1e-4, 2e-4]  0.9148
    # sgdm w/ bn:  [1e-3, 2e-3, 5e-3]  0.8982
    # sgdm w/o bn: [5e-3, 1e-2, 2e-2]  0.9177

    'hidden_layer_sizes': [(128, 128), (64, 64, 64)],
    'learning_rate': [5e-3, 1e-2, 2e-2],
}]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model_fn, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

from sklearn.svm import SVC

model_fn = functools.partial(
    SVC, random_state=SEED, probability=True,
    class_weight='balanced')
param_grid_list = [{
    'kernel': ['linear'],
    'C': [0.01, 0.1, 1],
}]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model_fn, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

from sklearn.neighbors import KNeighborsClassifier

model_fn = functools.partial(
    KNeighborsClassifier, n_jobs=-1, algorithm='auto', p=2)
param_grid_list = [{
    'n_neighbors': [3, 5, 7],
}]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model_fn, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

from sklearn.naive_bayes import GaussianNB

model_fn = GaussianNB
param_grid_list = [{}]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model_fn, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

# from sklearn.exceptions import ConvergenceWarning
# import warnings
# warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.linear_model import LogisticRegression

model_fn = functools.partial(
    LogisticRegression, random_state=SEED, n_jobs=-1, 
    max_iter=2000, solver='saga', penalty='l2',
    )
param_grid_list = [{
    'C': [1.0, 10.0, 100.0],
}]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(
    model_fn, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(
    best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

from sklearn.tree import DecisionTreeClassifier

model_fn = functools.partial(
    DecisionTreeClassifier, random_state=SEED,
    ccp_alpha=0.0, class_weight='balanced', max_depth=5)
param_grid_list = [{
        'criterion': ['gini', 'log_loss', 'entropy'],
    }
]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model_fn, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

from sklearn.ensemble import RandomForestClassifier

model_fn = functools.partial(
    RandomForestClassifier, random_state=SEED, n_jobs=-1,
    ccp_alpha=0.0, class_weight='balanced', max_depth=5, bootstrap=False)
param_grid_list = [{
        'criterion': ['gini', 'log_loss', 'entropy'],
        'n_estimators': [50, 100, 200],
    }
]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model_fn, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

from sklearn.ensemble import GradientBoostingClassifier

model = functools.partial(
    GradientBoostingClassifier, random_state=SEED,
    learning_rate=0.1, max_depth=5, loss='log_loss', 
    n_estimators=100)
param_grid_list = [{
        'criterion': ['friedman_mse', 'squared_error'],
    }
]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

from sklearn.ensemble import AdaBoostClassifier
model = functools.partial(AdaBoostClassifier, random_state=SEED, algorithm="SAMME")
param_grid_list = [{
    'n_estimators': [100, 200, 500],
    'learning_rate': [1.0],
}]
best_metrics, best_params, best_model = classification_dataset.perform_grid_search(model, param_grid_list)
metrics_internal, metrics_external = classification_dataset.evaluate_test_sets(best_model)
results_dict[get_model_name(model_fn)] = np.concatenate([best_metrics, metrics_internal, metrics_external], axis=0)

dataframe = pd.DataFrame(results_dict).transpose().reset_index()
dataframe.columns = ['model', 'val_auc', 'val_aupr', 'val_acc', 'intest_auc', 'intest_aupr', 'intest_acc', 'extest_auc', 'extest_aupr', 'extest_acc']
# dataframe.to_csv()
filename = os.path.join(RESULTS_DIR, encode_method, impute_method, f'{fs_method}{int(fs_ratio * 142)}.csv')
dirname = os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)
dataframe.to_csv(filename, index=False)