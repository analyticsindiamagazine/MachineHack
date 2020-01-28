#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import scipy
from sklearn.metrics import roc_curve, auc


# # Feature Generation for Train:

# In[ ]:


train = pd.read_csv("train_data.csv")
patient=pd.DataFrame(train.patient_id.unique())
patient.columns=['patient_id']

# Recency:
pat_event=train.groupby(['patient_id', 'event_name'])['event_time'].min().reset_index()
pat_event2=pat_event.pivot(index='patient_id', columns='event_name', values='event_time').reset_index()
pat_event2= pat_event2.rename_axis(None, axis=1)

del pat_event

pat_spcl=train.groupby(['patient_id', 'specialty'])['event_time'].min().reset_index()
pat_spcl2=pat_spcl.pivot(index='patient_id', columns='specialty', values='event_time').reset_index()
pat_spcl2= pat_spcl2.rename_axis(None, axis=1)

del pat_spcl

pat_pln=train.groupby(['patient_id', 'plan_type'])['event_time'].min().reset_index()
pat_pln2=pat_pln.pivot(index='patient_id', columns='plan_type', values='event_time').reset_index()
pat_pln2= pat_pln2.rename_axis(None, axis=1)

del pat_pln

pat_event2=pat_event2.add_prefix('recency__event_name__')
pat_event2 = pat_event2.rename(columns = {'recency__event_name__patient_id': "patient_id"})

pat_spcl2=pat_spcl2.add_prefix('recency__specialty__')
pat_spcl2 = pat_spcl2.rename(columns = {'recency__specialty__patient_id': "patient_id"})

pat_pln2=pat_pln2.add_prefix('recency__plan_type__')
pat_pln2 = pat_pln2.rename(columns = {'recency__plan_type__patient_id': "patient_id"})

patient = pd.merge(patient, pat_event2, on='patient_id')
patient = pd.merge(patient, pat_spcl2, on='patient_id')
patient = pd.merge(patient, pat_pln2, on='patient_id')

del pat_event2
del pat_spcl2
del pat_pln2


# Frequency:

for i in range(30,1110,30):
    dt=train[train['event_time']<=i].reset_index()
    del dt['index']
    dt1=dt.groupby(['patient_id', 'event_name'])['event_time'].count().reset_index()
    dt2=dt1.pivot(index='patient_id', columns='event_name', values='event_time').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('frequency__event_name__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'frequency__event_name__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
    
for i in range(30,1110,30):
    dt=train[train['event_time']<=i].reset_index()
    del dt['index']
    dt1=dt.groupby(['patient_id', 'specialty'])['event_time'].count().reset_index()
    dt2=dt1.pivot(index='patient_id', columns='specialty', values='event_time').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('frequency__specialty__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'frequency__specialty__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
    
for i in range(30,1110,30):
    dt=train[train['event_time']<=i].reset_index()
    del dt['index']
    dt1=dt.groupby(['patient_id', 'plan_type'])['event_time'].count().reset_index()
    dt2=dt1.pivot(index='patient_id', columns='plan_type', values='event_time').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('frequency__plan_type__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'frequency__plan_type__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
    
# NormChange:

for i in range(30,570,30):
    data_post = train[train['event_time']<=i].reset_index(drop=True)
    data_pre = train[train['event_time']>i].reset_index(drop=True)
    
    data_post1=data_post.groupby(['patient_id', 'event_name'])['event_time'].count().reset_index()
    data_post1['feature_value_post'] = data_post1['event_time']/i
    
    data_pre1=data_pre.groupby(['patient_id', 'event_name'])['event_time'].count().reset_index()
    data_pre1['feature_value_pre'] = data_pre1['event_time']/(1080 - i)
    
    normChange = pd.merge(data_post1, data_pre1, on=['patient_id', 'event_name'], how='outer')
    normChange.fillna(0, inplace=True)
    normChange['feature_value'] = np.where(normChange['feature_value_post']>normChange['feature_value_pre'], 1, 0)
    normChange=normChange[['patient_id','event_name','feature_value']]
    
    dt2=normChange.pivot(index='patient_id', columns='event_name', values='feature_value').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('normChange__event_name__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'normChange__event_name__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
    
for i in range(30,570,30):
    data_post = train[train['event_time']<=i].reset_index(drop=True)
    data_pre = train[train['event_time']>i].reset_index(drop=True)
    
    data_post1=data_post.groupby(['patient_id', 'specialty'])['event_time'].count().reset_index()
    data_post1['feature_value_post'] = data_post1['event_time']/i
    
    data_pre1=data_pre.groupby(['patient_id', 'specialty'])['event_time'].count().reset_index()
    data_pre1['feature_value_pre'] = data_pre1['event_time']/(1080 - i)
    
    normChange = pd.merge(data_post1, data_pre1, on=['patient_id', 'specialty'], how='outer')
    normChange.fillna(0, inplace=True)
    normChange['feature_value'] = np.where(normChange['feature_value_post']>normChange['feature_value_pre'], 1, 0)
    normChange=normChange[['patient_id','specialty','feature_value']]
    
    dt2=normChange.pivot(index='patient_id', columns='specialty', values='feature_value').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('normChange__specialty__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'normChange__specialty__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left') 
    
    
for i in range(30,570,30):
    data_post = train[train['event_time']<=i].reset_index(drop=True)
    data_pre = train[train['event_time']>i].reset_index(drop=True)
    
    data_post1=data_post.groupby(['patient_id', 'plan_type'])['event_time'].count().reset_index()
    data_post1['feature_value_post'] = data_post1['event_time']/i
    
    data_pre1=data_pre.groupby(['patient_id', 'plan_type'])['event_time'].count().reset_index()
    data_pre1['feature_value_pre'] = data_pre1['event_time']/(1080 - i)
    
    normChange = pd.merge(data_post1, data_pre1, on=['patient_id', 'plan_type'], how='outer')
    normChange.fillna(0, inplace=True)
    normChange['feature_value'] = np.where(normChange['feature_value_post']>normChange['feature_value_pre'], 1, 0)
    normChange=normChange[['patient_id','plan_type','feature_value']]
    
    dt2=normChange.pivot(index='patient_id', columns='plan_type', values='feature_value').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('normChange__plan_type__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'normChange__plan_type__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
patient.to_csv('train_features.csv', index=False)


# # Feature Generation for Test:

# In[ ]:


train = pd.read_csv("test_data.csv")
patient=pd.DataFrame(train.patient_id.unique())
patient.columns=['patient_id']

# Recency:

pat_event=train.groupby(['patient_id', 'event_name'])['event_time'].min().reset_index()
pat_event2=pat_event.pivot(index='patient_id', columns='event_name', values='event_time').reset_index()
pat_event2= pat_event2.rename_axis(None, axis=1)

del pat_event

pat_spcl=train.groupby(['patient_id', 'specialty'])['event_time'].min().reset_index()
pat_spcl2=pat_spcl.pivot(index='patient_id', columns='specialty', values='event_time').reset_index()
pat_spcl2= pat_spcl2.rename_axis(None, axis=1)

del pat_spcl

pat_pln=train.groupby(['patient_id', 'plan_type'])['event_time'].min().reset_index()
pat_pln2=pat_pln.pivot(index='patient_id', columns='plan_type', values='event_time').reset_index()
pat_pln2= pat_pln2.rename_axis(None, axis=1)

del pat_pln

pat_event2=pat_event2.add_prefix('recency__event_name__')
pat_event2 = pat_event2.rename(columns = {'recency__event_name__patient_id': "patient_id"})

pat_spcl2=pat_spcl2.add_prefix('recency__specialty__')
pat_spcl2 = pat_spcl2.rename(columns = {'recency__specialty__patient_id': "patient_id"})

pat_pln2=pat_pln2.add_prefix('recency__plan_type__')
pat_pln2 = pat_pln2.rename(columns = {'recency__plan_type__patient_id': "patient_id"})


patient = pd.merge(patient, pat_event2, on='patient_id')
patient = pd.merge(patient, pat_spcl2, on='patient_id')
patient = pd.merge(patient, pat_pln2, on='patient_id')

del pat_event2
del pat_spcl2
del pat_pln2

# Frequency:

for i in range(30,1110,30):
    dt=train[train['event_time']<=i].reset_index()
    del dt['index']
    dt1=dt.groupby(['patient_id', 'event_name'])['event_time'].count().reset_index()
    dt2=dt1.pivot(index='patient_id', columns='event_name', values='event_time').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('frequency__event_name__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'frequency__event_name__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
    
for i in range(30,1110,30):
    dt=train[train['event_time']<=i].reset_index()
    del dt['index']
    dt1=dt.groupby(['patient_id', 'specialty'])['event_time'].count().reset_index()
    dt2=dt1.pivot(index='patient_id', columns='specialty', values='event_time').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('frequency__specialty__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'frequency__specialty__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
    
for i in range(30,1110,30):
    dt=train[train['event_time']<=i].reset_index()
    del dt['index']
    dt1=dt.groupby(['patient_id', 'plan_type'])['event_time'].count().reset_index()
    dt2=dt1.pivot(index='patient_id', columns='plan_type', values='event_time').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('frequency__plan_type__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'frequency__plan_type__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')

    
# NormChange:

for i in range(30,570,30):
    data_post = train[train['event_time']<=i].reset_index(drop=True)
    data_pre = train[train['event_time']>i].reset_index(drop=True)
    
    data_post1=data_post.groupby(['patient_id', 'event_name'])['event_time'].count().reset_index()
    data_post1['feature_value_post'] = data_post1['event_time']/i
    
    data_pre1=data_pre.groupby(['patient_id', 'event_name'])['event_time'].count().reset_index()
    data_pre1['feature_value_pre'] = data_pre1['event_time']/(1080 - i)
    
    normChange = pd.merge(data_post1, data_pre1, on=['patient_id', 'event_name'], how='outer')
    normChange.fillna(0, inplace=True)
    normChange['feature_value'] = np.where(normChange['feature_value_post']>normChange['feature_value_pre'], 1, 0)
    normChange=normChange[['patient_id','event_name','feature_value']]
    
    dt2=normChange.pivot(index='patient_id', columns='event_name', values='feature_value').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('normChange__event_name__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'normChange__event_name__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
    
for i in range(30,570,30):
    data_post = train[train['event_time']<=i].reset_index(drop=True)
    data_pre = train[train['event_time']>i].reset_index(drop=True)
    
    data_post1=data_post.groupby(['patient_id', 'specialty'])['event_time'].count().reset_index()
    data_post1['feature_value_post'] = data_post1['event_time']/i
    
    data_pre1=data_pre.groupby(['patient_id', 'specialty'])['event_time'].count().reset_index()
    data_pre1['feature_value_pre'] = data_pre1['event_time']/(1080 - i)
    
    normChange = pd.merge(data_post1, data_pre1, on=['patient_id', 'specialty'], how='outer')
    normChange.fillna(0, inplace=True)
    normChange['feature_value'] = np.where(normChange['feature_value_post']>normChange['feature_value_pre'], 1, 0)
    normChange=normChange[['patient_id','specialty','feature_value']]
    
    dt2=normChange.pivot(index='patient_id', columns='specialty', values='feature_value').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('normChange__specialty__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'normChange__specialty__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left') 
    
    
for i in range(30,570,30):
    data_post = train[train['event_time']<=i].reset_index(drop=True)
    data_pre = train[train['event_time']>i].reset_index(drop=True)
    
    data_post1=data_post.groupby(['patient_id', 'plan_type'])['event_time'].count().reset_index()
    data_post1['feature_value_post'] = data_post1['event_time']/i
    
    data_pre1=data_pre.groupby(['patient_id', 'plan_type'])['event_time'].count().reset_index()
    data_pre1['feature_value_pre'] = data_pre1['event_time']/(1080 - i)
    
    normChange = pd.merge(data_post1, data_pre1, on=['patient_id', 'plan_type'], how='outer')
    normChange.fillna(0, inplace=True)
    normChange['feature_value'] = np.where(normChange['feature_value_post']>normChange['feature_value_pre'], 1, 0)
    normChange=normChange[['patient_id','plan_type','feature_value']]
    
    dt2=normChange.pivot(index='patient_id', columns='plan_type', values='feature_value').reset_index()
    dt2=dt2.rename_axis(None, axis=1)
    dt2.fillna(0,inplace=True)
    
    chk2=dt2.add_prefix('normChange__plan_type__')
    chk2=chk2.add_suffix('__'+str(i))
    chk2=chk2.rename(columns = {'normChange__plan_type__patient_id__'+str(i): 'patient_id'})
    patient=pd.merge(patient,chk2, on='patient_id', how='left')
    
patient.to_csv('test_features.csv', index=False)


# # Feature Selection:

# In[2]:


train_features = pd.read_csv("train_features.csv")
train_labels = pd.read_csv("train_labels.csv")
test_features = pd.read_csv("test_features.csv")

train_features1=pd.merge(train_features,train_labels, on='patient_id', how='left')
train_y=train_features1.outcome_flag

del train_features1['outcome_flag']
del train_features1['patient_id']
del test_features['patient_id']

train_features1.fillna(9999999, inplace=True)


# In[ ]:


X_train, X_validation, y_train, y_validation = train_test_split(train_features1, train_y, train_size=0.7, random_state=1234)
model1=lgb.LGBMClassifier(n_estimators=10000,
                          n_jobs= -1,
                          min_child_weight= 1,
                          feature_fraction= 0.5, #0.5
                          num_leaves= 80,#80
                          learning_rate= 0.1,#0.1
                          colsample_bytree=0.3, #New                     
                          random_state=1234)
model1.fit(train_features1, train_y, eval_set=[(X_validation, y_validation)],verbose=200,early_stopping_rounds=500)

t1=model1.feature_importances_
t1=pd.DataFrame(t1)

t2=train_features1.columns
t2=pd.DataFrame(t2)

t1.columns=['Importance']
t2.columns=['Variable']

t3=pd.concat([t2,t1],axis=1)
t4=t3[t3['Importance'] > 0]

my_cols = list(t4.Variable)

train_features2 = train_features1[my_cols]
test_features2 = test_features[my_cols]

test_features2.fillna(9999999, inplace=True)


# # Model Building:

# In[ ]:


folds = StratifiedKFold(n_splits=30, shuffle=True, random_state=12345678)
models = []
scores = []
i=1
dt_v1=pd.DataFrame()
for train_index, test_index in folds.split(train_features2, train_y):
    print('###########')
    X_train, X_val = train_features2.iloc[train_index], train_features2.iloc[test_index]
    y_train, y_val = train_y.iloc[train_index], train_y.iloc[test_index]
    lgb_params = {'n_estimators': 10000,
                  'n_jobs': -1,
                  'min_child_weight': 1,
                  'feature_fraction' : 0.7,
                  'num_leaves' : 40, 
                  'learning_rate':0.01,
                  'random_state':1234,
                  'seed':1234 
                  }
    model=lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train,eval_set=[(X_val, y_val)],verbose=200,eval_metric='auc',early_stopping_rounds=400)   
    scores.append([model.predict_proba(test_features2)])
    print('\n')
    
s_test=np.mean(scores, axis=0)
s_test=pd.DataFrame(s_test[0])
s_test=s_test.add_prefix("pred_")

test_features_patient = pd.read_csv("test_features.csv",usecols=["patient_id"])
s_test=pd.concat([test_features_patient,s_test],axis=1)

s_test=s_test[['patient_id','pred_1']]
s_test.columns=['patient_id','outcome_flag']
s_test.to_excel('submission_Best_score.xlsx', index=False)

