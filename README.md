# Credit Card Fraudulent Transaction Detection Model

## Overview
This repository contains a project focused on detecting fraudulent credit card transactions using machine learning. The dataset used for training and evaluation is highly imbalanced, with a significant majority of transactions being legitimate and a small fraction being fraudulent. The project employs data preprocessing, sampling techniques, and a Logistic Regression model to achieve effective fraud detection.

## Table of Contents
- [About the Dataset](#about-the-dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## About the Dataset
The dataset consists of credit card transactions with the following features:
- **Time**: Number of seconds elapsed between the transaction and the first transaction in the dataset.
- **V1 to V28**: Anonymized features resulting from a PCA transformation of the original features.
- **Amount**: Transaction amount in USD.
- **Class**: Label indicating whether the transaction is fraudulent (1) or legitimate (0).

The dataset contains 284,807 transactions, with 492 fraudulent transactions.

## Project Structure
The project is organized as follows:
- `data/`: Directory containing the dataset.
- `notebooks/`: Jupyter notebooks used for data exploration and model training.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `README.md`: Project documentation.

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/codewithshivam01/Credit-Card-Fraudulant-Transaction-Detection-Model.git
    cd Credit-Card-Fraudulent-Transaction-Detection-Model
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv env
    source env/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Data Loading and Exploration:
Load the dataset and explore its structure and statistical properties.
```python
import pandas as pd

credit_card_data = pd.read_csv('data/creditcard.csv')
print(credit_card_data.head())
```


## Data Preprocessing:
Check for missing values and handle them if any. Separate the data into legitimate (legit) and fraudulent (fraud) transactions. Balance the dataset by sampling an equal number of legitimate transactions.
```python
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

legit_sample = legit.sample(n=492)
new_df = pd.concat([legit_sample, fraud], axis=0)
```
 ## Model Training and Evaluation:
 Split the data into training and test sets. Train a Logistic Regression model. Evaluate the model's performance on both training and test sets.
 ```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = new_df.drop(columns='Class', axis=1)
Y = new_df['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
```

## Results
* Accuracy on Training data: 93.77%
* Accuracy score on Test Data: 89.85%
