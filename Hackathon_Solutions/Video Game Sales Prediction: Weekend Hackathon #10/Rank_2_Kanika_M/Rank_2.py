#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 03:36:15 2020

@author: kanikamiglani
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingRegressor
#from sklearn.ensemble import StackingRegressor

data1 = pd.read_csv('train_modified.csv')
data3 = pd.read_csv('Train.csv')
data2 = pd.read_csv('test_modified.csv')

"""
print(min(data1['CRITICS_POINTS']))
print(max(data1['CRITICS_POINTS']))
print(min(data1['USER_POINTS']))
print(max(data1['USER_POINTS']))

plt.figure(figsize=(9,6))
sns.heatmap(data3.corr(),annot=True,linewidth = 0.5, cmap='coolwarm')

print(data1.head())
print(data1.shape)
print(data1.info())
print(data1.dtypes)
print(data1.describe())
print(data1.columns.tolist())
print(data1.isnull().sum())
print(data2.isnull().sum())

print(data1['ID'].unique())
print(data1['ID'].value_counts())
print(data2['ID'].unique())
print(data2['ID'].value_counts())
#print(data1['ID'].value_counts(normalize=True)*100)
#sns.countplot(data1['ID'])

print(data1['CONSOLE'].unique())
print(data1['CONSOLE'].value_counts())
print(data2['CONSOLE'].unique())
print(data2['CONSOLE'].value_counts())
#print(data1['CONSOLE'].value_counts(normalize=True)*100)
#sns.countplot(data1['CONSOLE'])

print(data1['YEAR'].unique())
print(data1['YEAR'].value_counts())
print(data2['YEAR'].unique())
print(data2['YEAR'].value_counts())
#print(data1['YEAR'].value_counts(normalize=True)*100)
#sns.countplot(data1['YEAR'])

print(data1['CATEGORY'].unique())
print(data1['CATEGORY'].value_counts())
print(data2['CATEGORY'].unique())
print(data2['CATEGORY'].value_counts())
#print(data1['CATEGORY'].value_counts(normalize=True)*100)
#sns.countplot(data1['CATEGORY'])

print(data1['PUBLISHER'].unique())
print(data1['PUBLISHER'].value_counts())
print(data2['PUBLISHER'].unique())
print(data2['PUBLISHER'].value_counts())
#print(data1['PUBLISHER'].value_counts(normalize=True)*100)
#sns.countplot(data1['PUBLISHER'])

print(data1['RATING'].unique())
print(data1['RATING'].value_counts())
print(data2['RATING'].unique())
print(data2['RATING'].value_counts())
#print(data1['RATING'].value_counts(normalize=True)*100)
#sns.countplot(data1['RATING'])

print(data1['CRITICS_POINTS'].unique())
print(data1['CRITICS_POINTS'].value_counts())
print(data2['CRITICS_POINTS'].unique())
print(data2['CRITICS_POINTS'].value_counts())
#print(data1['CRITICS_POINTS'].value_counts(normalize=True)*100)
#sns.countplot(data1['CRITICS_POINTS'])

print(data1['USER_POINTS'].unique())
print(data1['USER_POINTS'].value_counts())
print(data2['USER_POINTS'].unique())
print(data2['USER_POINTS'].value_counts())
#print(data1['USER_POINTS'].value_counts(normalize=True)*100)
#sns.countplot(data1['USER_POINTS'])

print(data1['SalesInMillions'].unique())
print(data1['SalesInMillions'].value_counts())

#print(data1['SalesInMillions'].value_counts(normalize=True)*100)
#sns.countplot(data1['SalesInMillions'])

plt.figure(figsize=(9,6))
sns.heatmap(data1.corr(),annot=True,linewidth = 0.5, cmap='coolwarm')

#X = pd.get_dummies(X)

#data1=pd.get_dummies(data1)
#data2=pd.get_dummies(data2)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,3,4,5])], remainder='passthrough')
X = ct.fit_transform(X)
data2 = ct.fit_transform(data2)
print(X.shape)
print(data2.shape)
"""

#data1 = data1.drop("ID", 1)
#data2 = data2.drop("ID", 1)

#u = (data1['USER_POINTS'])*(data1['USER_POINTS'])
#data1['USER_POINTS'] = u

#data1 = data1.drop("CRITICS_POINTS", 1)
#data2 = data2.drop("CRITICS_POINTS", 1)

#data1 = data1.drop("YEAR", 1)
#data2 = data2.drop("YEAR", 1)

X=data1.drop("SalesInMillions",1)
y=data1[["SalesInMillions"]]

#print(data1[['CRITICS_POINTS', 'USER_POINTS']].corr())
#print(data1[['CRITICS_POINTS', 'YEAR']].corr())
#print(data1[['YEAR', 'USER_POINTS']].corr())

#"""
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
regressor1 = HistGradientBoostingRegressor()
#RMSE: 1.629386

import lightgbm as lgb
from lightgbm import LGBMRegressor
regressor2 = LGBMRegressor(random_state=0)
#RMSE: 1.673168

from catboost import CatBoostRegressor
regressor3 = CatBoostRegressor(iterations=2000, random_state = 0, verbose = 200)

import xgboost as xgb
from xgboost import XGBRegressor
regressor4 = xgb.XGBRegressor()

from nimbusml.ensemble import LightGbmRegressor
regressor5 = LightGbmRegressor(random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor6 = RandomForestRegressor(n_estimators = 1000, random_state = 0)

from sklearn.ensemble import GradientBoostingRegressor
regressor7 = GradientBoostingRegressor(random_state=0)

from sklearn import linear_model
regressor8 = linear_model.BayesianRidge()

from sklearn.svm import SVR
regressor9 = SVR(kernel = 'rbf')

from sklearn.neural_network import MLPRegressor
regressor10 = MLPRegressor(random_state=0, max_iter=1000)

from sklearn.ensemble import ExtraTreesRegressor
regressor11 = ExtraTreesRegressor(n_estimators=1000, random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor12 = DecisionTreeRegressor(random_state = 0)

#estimators = [('hist', regressor1), ('lgbm', regressor2), ('cb', regressor3), ('xgb', regressor4), ('nimbus', regressor5), ('gbr', regressor7), ('br', regressor8), ('svr', regressor9)]
estimator = [('hist', regressor1), ('lgbm', regressor2), ('cb', regressor3), 
             ('xgb', regressor4), ('nimbus', regressor5), ('rfr', regressor6), 
             ('gbr', regressor7), ('br', regressor8), ('svr', regressor9)]
weight = [4,3,1,1,1,1,1,1,1]
regressor = VotingRegressor(estimators = estimator, weights = weight)

#regressor = StackingRegressor(estimators=estimators,final_estimator=vregressor)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

r = r2_score(y_test, y_pred)
print("R^2: %f" % (r*100))

m = mean_squared_error(y_test, y_pred)
print("MSE: %f" % (m))

rmse = np.sqrt(m)
print("RMSE: %f" % (rmse))

#accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5)
#print("K FOlD Accuracy: {:.2f} %".format(accuracies.mean()*100))
#print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

data2 = sc.transform(data2)
y_p = regressor.predict(data2)
df =  pd.DataFrame(y_p, index= None)
df.columns = ["SalesInMillions"]
df.to_excel("Sample_Submission.xlsx", index = None)
df.to_csv("Sample_Submission.csv", index = None)
#"""

"""
from nimbusml.ensemble import LightGbmRegressor
#regressor = LightGbmRegressor(random_state=0)
#RMSE: 1.634116

from catboost import CatBoostRegressor
regressor = CatBoostRegressor(iterations=2000, random_state = 0, verbose = 200)
#RMSE: 1.733194

import xgboost as xgb
from xgboost import XGBRegressor
#regressor = xgb.XGBRegressor()
#RMSE: 1.769113

from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
#RMSE: 1.750798

from sklearn.ensemble import GradientBoostingRegressor
#regressor = GradientBoostingRegressor(random_state=0)
#RMSE: 1.805883

from sklearn.tree import DecisionTreeRegressor
#regressor = DecisionTreeRegressor(random_state = 0)
#RMSE: 2.689125

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
#RMSE: 2.006418

from sklearn.ensemble import AdaBoostRegressor
#regressor = AdaBoostRegressor(random_state=0, n_estimators=1000)
#RMSE: 3.150858

from sklearn.ensemble import ExtraTreesRegressor
#regressor = ExtraTreesRegressor(n_estimators=500, random_state=0)
#RMSE: 2.710881

from sklearn.neural_network import MLPRegressor
#regressor = MLPRegressor(random_state=0, max_iter=1000)
#RMSE: 2.529909

from sklearn import linear_model
#regressor = linear_model.BayesianRidge()
#RMSE: 2.000902

from sklearn.linear_model import PassiveAggressiveRegressor
#regressor = PassiveAggressiveRegressor(max_iter=1000, random_state=0,tol=1e-3)
#RMSE: 3.030982

from sklearn.linear_model import Ridge
#regressor = Ridge(alpha=1.0)
#RMSE: 2.017923
"""
