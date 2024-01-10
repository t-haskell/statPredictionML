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


import xgboost

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
X = merged_df[['Position', 'Games Played', 'Shots', 'Shots On Goal', '2pt Shots', '2pt Shots On Goal', 'Groundballs', 'Caused Turnovers']].values
print(X[0:8])
le_pos = preprocessing.LabelEncoder()
le_pos.fit(['FO', 'M', 'SSDM', 'LSM', 'A', 'D', 'G','nan'])
X[:,0] = le_pos.transform(X[:,0])
X = X.astype(float)
print(X.shape)
print(X[0:8])
# selecting dependent variable (target variable)
y = merged_df["Points"]
print(y.shape)
print(y[0:8])




