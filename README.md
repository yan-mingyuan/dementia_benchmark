# Dementia Benchmark

## Introduction
The Dementia Benchmark project aims to establish a standardized benchmark for dementia prediction models. It encompasses functionalities for data preprocessing, model training, and evaluation.

## Requirements
- numpy==1.26.4
- pandas==2.2.1
- scikit_learn==1.4.0

## File Structure
- `resources/`: Contains the dataset files.
- `config.py`: Defines constant variables used throughout the project.
- `utils.py`: Houses utility functions utilized in various aspects of the project.
- `dataset.py`: Implements functionalities to manage the dataset for model evaluation, including internal and external validation.
- `1_preprocess.ipynb`: A Jupyter Notebook for data preprocessing tasks such as table projection, table joining, and variable encoding.
- `2_eval.ipynb`: A Jupyter Notebook containing the main pipeline for model training, hyperparameter tuning, and evaluation.

## Usage
1. Execute `1_preprocess.ipynb` to preprocess the dataset.
2. Run `2_eval.ipynb` to train models, perform hyperparameter tuning, and evaluate model performance.