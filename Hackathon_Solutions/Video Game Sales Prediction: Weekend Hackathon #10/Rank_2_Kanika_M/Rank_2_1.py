#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:59:38 2020

@author: kanikamiglani
"""

import pandas as pd
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

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
print(train.shape)
print(test.shape)
print(data.shape)
print(data.apply(lambda x: sum(x.isnull())))
print(data.apply(lambda x: len(x.unique())))

categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns if x not in ['source']]

data['RATING']= data['RATING'].replace('RP', 'M', regex=True)
data['RATING']= data['RATING'].replace('K-A', 'M', regex=True)
data['RATING']= data['RATING'].replace('AO', 'M', regex=True)

years = 2020-data['YEAR']
data['YEAR']= years

for col in categorical_columns:
    print('\nFrequency of Categories for varible %s'%col)
    print(data[col].value_counts())
    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['ID'] = le.fit_transform(data['ID'])

data = pd.get_dummies(data, columns=['CONSOLE','CATEGORY','PUBLISHER','RATING'])
print(data.dtypes)

train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

test.drop(['SalesInMillions','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)
