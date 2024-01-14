# Machine Learning Lacrosse Stat Prediction - Revamped

**Created by:** Thomas Haskell

## Description

This project involves recreating the 2023 summer project to better utilize machine learning techniques discovered during the IBM ML with Python Certification. The file 'revamp.py' serves as a replacement for the original 'main.py'.

The project compares the results and error metrics of two models on the PLL dataset. One model utilizes a multiple linear regression algorithm from sklearn, while the other employs XGBoost's gradient descent algorithm.

## Datasets

- 2023: [Link to Dataset](stats2023.csv)
- 2022: [Link to Dataset](stats2022.csv)
- 2021: [Link to Dataset](stats2021.csv)
- 2020: [Link to Dataset](stats2020.csv)
- 2019: [Link to Dataset](stats2019.csv)
- 2018: [Link to Dataset](stats2018.csv)

## Preprocessing

The datasets are merged, and feature selection is performed. The independent variables include 'Position', 'Games Played', 'Shots On Goal', '2pt Shots', '2pt Shots On Goal', 'Groundballs', and 'Caused Turnovers'. Label encoding is applied to the 'Position' variable, and the target variable is 'Points'. The data is split into training and testing sets.

## Data Visualization

A heatmap is used to visualize the correlation between features in the training dataset.
![pllFeatureCorrelation](https://github.com/t-haskell/statPredictionPLL/assets/94083215/24710b95-caee-4b2d-bed7-94c82334b8c5)
![pllTrainingCorrelation](https://github.com/t-haskell/statPredictionPLL/assets/94083215/87c6e033-6c63-4bb3-aacc-cb9a37d274cf)



## Models and Evaluation Metrics

### 1. Standard Linear Regression

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R2 Score

### 2. XGBoost Regression

- Mean Squared Error (MSE)
- R2 Score

### 3. Optimized XGBoost Regression

- Mean Squared Error (MSE)
- R2 Score

Hyperparameters are adjusted using GridSearchCV and RandomizedSearchCV.

## Implementation

The project script, 'revamp.py', implements each model, trains the models, makes predictions, and evaluates the results using the specified metrics. The performance of the revamped project is highlighted, demonstrating improved results compared to the original.

This project showcases practical experience in applying machine learning techniques to sports statistics, enhancing predictive modeling and evaluation metrics.
