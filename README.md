# Credit Card Fraudulent Transaction Detection Model

## Overview
This repository contains a project focused on detecting fraudulent credit card transactions using machine learning. The dataset used for training and evaluation is highly imbalanced, with a significant majority of transactions being legitimate and a small fraction being fraudulent. The project employs data preprocessing, sampling techniques, and a Logistic Regression model to achieve effective fraud detection.

## Table of Contents
- [About the Dataset](#about-the-dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

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
    git clone https://github.com/your-username/Credit-Card-Fraudulent-Transaction-Detection-Model.git
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
