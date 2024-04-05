# Dementia Benchmark

## Introduction
The Dementia Benchmark project aims to establish a standardized benchmark for dementia prediction models. It encompasses functionalities for data preprocessing, model training, and evaluation.

## File Structure
- `resources/`: Contains the dataset files.
- `config.py`: Defines constant variables used throughout the project.
- `utilities/`: Houses utility functions utilized in various aspects of the project.
- `checkpoints/`: These checkpoints store the states of the imputer, feature selector, and predictor. As retraining is time-consuming and resource-intensive, caching checkpoints significantly accelerate runtime speeds.
- `results/`: Stores evaluation results.
- `dataset.py`: Implements functionalities to manage the dataset for model evaluation, including internal and external validation.
- `0_preprocess.ipynb`: A Jupyter Notebook for data preprocessing tasks such as table projection, table joining, and variable encoding.
- `eval_{enc}_{imp}_{fs}.ipynb`: A Jupyter Notebook containing the main pipeline for model training, hyperparameter tuning, and evaluation. The encode_method, impute_method, and fs_method used are included in the filename.

## Requirements
lightgbm==4.3.0
numpy==1.26.4
pandas==2.2.1
scikit_learn==1.4.1.post1
scipy==1.13.0
torch==2.0.1
torch_geometric==2.5.2
tqdm==4.65.1
xgboost==2.0.3

## Setup
```bash
conda create -n dementia python=3.10
conda activate dementia
pip install -r requirements.txt
```

## Usage
1. Execute `0_preprocess.ipynb` to preprocess the dataset.
2. Run `eval_{enc}_{imp}_{fs}.ipynb` to train models, perform hyperparameter tuning, and evaluate model performance. The encode_method, impute_method, and fs_method used are specified in the filename.