

# Credit-Card-Fraudulent-Transaction-Detection-Model

## Overview

This repository contains the implementation of a machine learning model designed to detect fraudulent credit card transactions. The goal of this project is to build a robust model that can accurately identify potentially fraudulent transactions from a dataset of credit card transactions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders.

- The dataset presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions.
- It is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.
- Features V1, V2, ..., V28 are the result of a PCA transformation. The only features which have not been transformed with PCA are `Time` and `Amount`.

## Installation

To use this repository, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Credit-Card-Fraudulent-Transaction-Detection-Model.git
    cd Credit-Card-Fraudulent-Transaction-Detection-Model
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model and make predictions, follow these steps:

1. **Preprocess the Data**: Run the preprocessing script to prepare the data for training.
    ```bash
    python preprocess.py
    ```

2. **Train the Model**: Train the machine learning model on the preprocessed data.
    ```bash
    python train.py
    ```

3. **Evaluate the Model**: Evaluate the trained model using various metrics.
    ```bash
    python evaluate.py
    ```

4. **Make Predictions**: Use the trained model to make predictions on new data.
    ```bash
    python predict.py
    ```

## Model

The model is built using the following machine learning algorithms:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Neural Networks

The final model is selected based on its performance on the evaluation metrics.

## Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Due to the imbalance in the dataset, precision, recall, and F1-score are considered more important than accuracy.

## Results

The final model achieved the following results on the test set:

- Precision: X.XX
- Recall: X.XX
- F1-Score: X.XX
- ROC-AUC: X.XX

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or pull requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

