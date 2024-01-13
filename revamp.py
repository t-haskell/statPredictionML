'''
Create by: Thomas Haskell


===== Machine Learning Lacrosse Stat Predicition - Revamped =====


DESCRIPTION: Recreating 2023 summer project to better utilize machine learning
techinques discovered from the IBM ML with Python Certification


'''
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn import metrics


import xgboost as xgb

f23 = pd.read_csv('stats2023.csv')
f22 = pd.read_csv('stats2022.csv')
f21 = pd.read_csv('stats2021.csv')
f20 = pd.read_csv('stats2020.csv')
f19 = pd.read_csv('stats2019.csv')
f18 = pd.read_csv('stats2018.csv')
print("2023: ", f23.describe(), f23.shape)
print("2022: ", f22.describe(), f22.shape)
# print("2021: ", f21.head(), f21.shape)
# print("2020: ", f20.head(), f20.shape)
# print("2019: ", f19.head(), f19.shape)
# print("2018: ", f18.head(), f18.shape)
years = [f23, f22, f21, f20, f19, f18]


## Preprocessing dataframes ##
merged_df = pd.concat([f23, f22, f21, f20, f19, f18], axis=0, join='outer')
# selecting independent variables (Feature Selection)
X = merged_df[['Position', 'Games Played', 'Shots On Goal', '2pt Shots', '2pt Shots On Goal', 'Groundballs', 'Caused Turnovers']].values
print(X[0:7])
le_pos = preprocessing.LabelEncoder()
le_pos.fit(['FO', 'M', 'SSDM', 'LSM', 'A', 'D', 'G','nan'])
X[:,0] = le_pos.transform(X[:,0])
X = X.astype(float)
print(X.shape)
print(X[0:7])
# selecting dependent variable (target variable)
y = merged_df["Points"].astype(float)
print(y.shape)
print(y[0:7])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

## Data Visualization ##
import seaborn as sb
import matplotlib.pyplot as plt
# Visualize the training data
train_df = pd.DataFrame(x_train, columns=['Position', 'Games Played', 'Shots On Goal', '2pt Shots', '2pt Shots On Goal', 'Groundballs', 'Caused Turnovers'])
train_df['Points'] = y_train.values
print(train_df.info())
print(train_df.head())
#sb.pairplot(train_df[0:15])
#sb.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
#plt.title('Training Feature Correlation')
#plt.show()

## Standard Linear Regression ##
from sklearn.linear_model import LinearRegression
LinearReg = LinearRegression()
x = np.asanyarray(x_train)
y = np.asanyarray(y_train)
LinearReg.fit(x, y) # training model on train sett
predictions = LinearReg.predict(x_test) # predicting with trained model on test sett
# Evaluating results
lr_MAE = np.mean(np.abs(predictions - y_test))
lr_MSE = np.mean((predictions - y_test) ** 2)
lr_R2 = metrics.r2_score(y_test, predictions)
# Showing results in tabular format as a data frame
Report = pd.DataFrame({'Model' : ['Standard Linear Regression'], 'MAE': [lr_MAE], 'MSE': [lr_MSE], 'R2': [lr_R2]})
print(Report)

## Using XGBoost training algorithms ##
# creating XGBoost object
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=0)
###### ADD NORMALIZATION TO DATA FOR XGB #########################
xgb_model.fit(x, y) # fitting model (training) on train data set
predictions = xgb_model.predict(x_test) # prediction
# Evaluate predictions
xg_mse = metrics.mean_squared_error(y_test, predictions, squared=True) # mean squared error
xg_R2 = metrics.r2_score(y_test, predictions)
print("XGBoost model MSE: ", xg_mse)
print("XGBoost model R2: ", xg_R2)


## Adjusting hyperparameters with GridSearchCV ##
from sklearn.model_selection import GridSearchCV
param_gridXG = {
    'max_depth' : [3,4,5],
    'learning_rate' : [0.5, 0.1, 0.001],
    'n_estimators' : [50, 100, 500],
    'min_child_weight': [1, 5]
}

## -----> Section below is commented out to reduce computational stress,
# since optimum parameters were printed and now used in new model below

# # Create GridSearchCV object (Commented out to avoid unnecessary computation)
# grid_searchXG = GridSearchCV(estimator=xgb_model, param_grid=param_gridXG, cv=5)

# # Fit grid search to training data (Commented out to avoid unnecessary computation)
# grid_searchXG.fit(x_train, y_train)

# # print optimal parameters (Commented out to avoid unnecessary computation)
# print("Optimum parameters for XGBoost model: ", grid_searchXG.best_params_)

# ## Predicting with new parameters
# xgb_optim = xgb.XGBRegressor(objective='reg:squarederror', seed=0, n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight= 1)
# xgb_optim.fit(x, y)
# predictions = xgb_optim.predict(x_test)
# # Evaluate predictions
# xg_mse = metrics.mean_squared_error(y_test, predictions, squared=True) # mean squared error
# xg_R2 = metrics.r2_score(y_test, predictions)
# print("Optimized XGBoost model MSE: ", xg_mse)
# print("Optimized XGBoost model R2: ", xg_R2)

## Adjusting parameters with Random Search ##
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 500, 1000],
    'min_child_weight': [1, 5],
    'gamma': [0.1, 0.5, 1],
    'subsample': [0.5, 0.75, 1],
    'colsample_bytree': [0.5, 0.75, 1]
}
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, n_iter=100, cv=5, random_state=42 )
random_search.fit(x, y)
print(random_search.best_params_)
# Predicting with new parameters
xgb_optim = xgb.XGBRegressor(objective='reg:squarederror', seed=0, n_estimators=1000, learning_rate=0.01, max_depth=3, min_child_weight= 5, subsample=0.5, gamma=0.1, colsample_bytree=1)
xgb_optim.fit(x, y)
predictions = xgb_optim.predict(x_test)
#Evaluate predictions
xg_mse = metrics.mean_squared_error(y_test, predictions, squared=True) # mean squared error
xg_R2 = metrics.r2_score(y_test, predictions)
print("Optimized XGBoost model MSE: ", xg_mse)
print("Optimized XGBoost model R2: ", xg_R2)



