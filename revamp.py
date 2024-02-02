'''
Create by: Thomas Haskell


===== Machine Learning Lacrosse Stat Predicition - Revamped =====


DESCRIPTION: Recreating 2023 summer project to better utilize machine learning
techinques discovered from the IBM ML with Python Certification. This file,
'revamp.py', is a replacement for the original, 'main.py'. 

-> Compares the results/error metrics for two models on the PLL dataset,
one using a multiply linear regression algorithms from sklearn and the other
using XGBoost's gradient decent algorithms. 


'''
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn import metrics
import xgboost as xgb


## Import Datasets ##
f23 = pd.read_csv('stats2023.csv')
f22 = pd.read_csv('stats2022.csv')
f21 = pd.read_csv('stats2021.csv')
f20 = pd.read_csv('stats2020.csv')
f19 = pd.read_csv('stats2019.csv')
f18 = pd.read_csv('stats2018.csv')
print("2023: ", f23.describe(), f23.shape)
print("2022: ", f22.describe(), f22.shape)


## Preprocessing Dataframes ##
merged_df = pd.concat([f23, f22, f21, f20, f19, f18], axis=0, join='outer')
# selecting independent variables (Feature Selection)
X = merged_df[['Position', 'Games Played', 'Shots On Goal', '2pt Shots', 'Power Play Shots', 'Groundballs', 'Turnovers']].values
print("\nIndependent Variable Samples:")
print(X[0:7])
le_pos = preprocessing.LabelEncoder()
le_pos.fit(['FO', 'M', 'SSDM', 'LSM', 'A', 'D', 'G','nan'])
X[:,0] = le_pos.transform(X[:,0])
X = X.astype(float)
print("\nIndependent Data w/ Position Labels Encoded:")
print(X.shape)
print(X[0:7])
# selecting dependent variable (target variable)
y = merged_df["Points"].astype(float)
print("\nDependent Variable Sample (Points):")
print(y.shape)
print(y[0:7])

# Seperating Test/Train Datasets 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

## Data Visualization ##
import seaborn as sb
import matplotlib.pyplot as plt
valid_features = merged_df.drop(columns=['First Name', 'Last Name', 'Jersey', 'Position', 'Team','Scores Against', 'Scores Against Average', 'Save Pct', 'Faceoff Wins', 'Saves', 'Short Handed Shots', 'Short Handed Goals Against', 'Power Play Goals Against', '2pt Goals Against', '2pt GAA','Faceoff Losses', 'Power Play Goals']).astype(float)
all_var_correlation = sb.heatmap(valid_features.corr(), annot=False, cmap='rocket')
all_var_correlation.set_xticklabels(all_var_correlation.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=8)
all_var_correlation.set_yticklabels(all_var_correlation.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=9)
plt.title('Heatmap for All Feature Correlation')
plt.show()
# Visualize the training data
train_df = pd.DataFrame(x_train, columns=['Position', 'Games Played', 'Shots On Goal', '2pt Shots', 'Power Play Shots', 'Groundballs', 'Turnovers'])
train_df['Points'] = y_train.values
print("\nVerifying Structure of Training Set:")
print(train_df.info())
sb.heatmap(train_df.corr(), annot=True, cmap='rocket')
plt.title('Training Feature Correlation')
plt.show()



## Standard Linear Regression ##
from sklearn.linear_model import LinearRegression
LinearReg = LinearRegression()
x = np.asanyarray(x_train)
y = np.asanyarray(y_train)
LinearReg.fit(x, y) # training model on train set
predictions = LinearReg.predict(x_test) # predicting with trained model on test set
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
xgb_model.fit(x, y) # fitting model (training) on train data set
predictions = xgb_model.predict(x_test) # prediction
# Evaluate predictions
xg_mse = metrics.mean_squared_error(y_test, predictions, squared=True) # mean squared error
xg_R2 = metrics.r2_score(y_test, predictions)
Report[1] = {'Model' : ['XGBoost'], 'MAE': [metrics.mean_absolute_error(y_test,predictions)], 'MSE': [xg_mse], 'R2': [xg_R2]}
print(Report)
print("XGBoost model MSE: ", xg_mse)
print("XGBoost model R2: ", xg_R2)


## GridSearchCV - Hyperparameter Tuning ##
from sklearn.model_selection import GridSearchCV
param_gridXG = {
    'max_depth' : [3,4,5],
    'learning_rate' : [0.5, 0.1, 0.001],
    'n_estimators' : [50, 100, 500],
    'min_child_weight': [1, 5]
}

## Optimal Param's hard coded in -----> Section below is commented out to reduce
# computational stress, since optimum parameters were printed and now used in new model below
'''
# Create GridSearchCV object (Commented out to avoid unnecessary computation)
grid_searchXG = GridSearchCV(estimator=xgb_model, param_grid=param_gridXG, cv=5)

# Fit grid search to training data (Commented out to avoid unnecessary computation)
grid_searchXG.fit(x_train, y_train)

# print optimal parameters (Commented out to avoid unnecessary computation)
print("Optimum parameters for XGBoost model: ", grid_searchXG.best_params_)
'''
## Predicting with new parameters
xgb_optim = xgb.XGBRegressor(objective='reg:squarederror', seed=0, n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight= 1)
xgb_optim.fit(x, y)
predictions = xgb_optim.predict(x_test)
# Evaluate predictions
xg_mse = metrics.mean_squared_error(y_test, predictions, squared=True) # mean squared error
xg_R2 = metrics.r2_score(y_test, predictions)
print("GridSearchCV Optimized XGBoost model MSE: ", xg_mse)
print("GridSearchCV Optimized XGBoost model R2: ", xg_R2)


## Random Search - Hyperparameter Tuning ##
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
############# Below is commented out to reduce compute time #########################
''' (Already found from running)
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, n_iter=100, cv=5, random_state=42 )
random_search.fit(x, y)
print(random_search.best_params_)
''' 
# Predicting with new parameters
xgb_optim = xgb.XGBRegressor(objective='reg:squarederror', seed=0, n_estimators=1000, learning_rate=0.01, max_depth=3, min_child_weight= 5, subsample=0.5, gamma=0.1, colsample_bytree=1)
xgb_optim.fit(x, y)
predictions = xgb_optim.predict(x_test)
#Evaluate predictions
xg_mse = metrics.mean_squared_error(y_test, predictions, squared=True) # mean squared error
xg_R2 = metrics.r2_score(y_test, predictions)
print("RandomizedSearchCV XGBoost model MSE: ", xg_mse)
print("RandomizedSearchCV XGBoost model R2: ", xg_R2)




## TensorFlow ##
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt

def create_tfNueralModel(hp):
    # Defining simple neural network
    model = keras.Sequential()
    # Layer 1
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(X.shape[1],)))
    # Layer 2
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    # Layer 3 (output layer)
    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size"), metrics=[keras.metrics.MeanAbsoluteError()])
    return model

## CREATING TUNER AND PERFORMING HYPERTUNE - Creating Hyperband object from Keras Tuning module, finding optimum hyperparameters
tuner = kt.Hyperband(create_tfNueralModel, objective=kt.Objective("val_mean_absolute_error", "min"), max_epochs=10, factor=5, directory='PLLproject', project_name='revamp')

# Stops search when val-loss stops improving for 10 epochs
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# Reduces learning rate of optimizer when val-loss stops improving for 5 epochs
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
# Saves the best model based on validation loss
model_checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Searching for optimal hyperparameters, similar to the fit() method but with an early stop arg
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[stop_early, reduce_lr, model_checkpoint])
# Retrieving the best model based off minimum MSE
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

## TRAINING MODEL - Finding optimal number of epochs with these hyperparameters
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)
maePerEpoch = history.history['mean_absolute_error']
best_epoch = maePerEpoch.index(min(maePerEpoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# Evaluating nueral network on the test set
nn_hypermodel = tuner.hypermodel.build(best_hps)
nn_hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)
eval_results = nn_hypermodel.evaluate(x_test, y_test)
print("Test MSE: ", eval_results[0])
print("Test MAE: ", eval_results[1])
predictions = nn_hypermodel.predict(x_test)
r2 = metrics.r2_score(y_test, predictions)
print("Test R2: ", r2)

