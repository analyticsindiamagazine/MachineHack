"""
Created on Sat Aug 22 21:20:42 2020

@author: Salim
"""


import pandas as pd 
import numpy as np
import prince
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

def my_groupby(df,primary_keys,dictionary_ops,renaming_dict):
    return df.groupby(primary_keys).agg(dictionary_ops).reset_index().rename(columns=renaming_dict)


def data_left_join(df1,df2,primary_key):
    return df1.merge(df2,how='left',on=primary_key)


def updated_df(df,primary_key,operation,columns):
    for cols in columns:
        df       = data_left_join(df,
                                   my_groupby(df,
                                              [primary_key],
                                              {cols:operation},
                                              {cols:primary_key+'_'+operation+'_'+cols}),
                                   primary_key)

    return df


traindf       = pd.read_csv(r'C:\Users\Salim\Downloads\Participants_Data\Train.csv')
testdf        = pd.read_csv(r'C:\Users\Salim\Downloads\Participants_Data\Test.csv')
train_y       = traindf['Class'].values
traindf       = traindf.drop(['Class'],axis=1)    


merged_data                          = pd.concat((traindf,testdf),axis=0)

categorical_cols                     = ['Area_Code','Locality_Code','Region_Code','Species']


mca                          = prince.MCA(n_components=2,random_state=202020).fit(merged_data[categorical_cols])
traindf.loc[:,'MCA1']        = mca.transform(traindf[categorical_cols])[0]
testdf.loc[:,'MCA1']         = mca.transform(testdf[categorical_cols])[0]



numerical_cols               = ['Height','Diameter']
pca                          = PCA(n_components=2,random_state=202020).fit(merged_data[numerical_cols])
traindf.loc[:,'PCA1']        = pca.transform(traindf[numerical_cols])[:,0]
testdf.loc[:,'PCA1']         = pca.transform(testdf[numerical_cols])[:,0]


merged_data                          = pd.concat([traindf,testdf],axis=0)

merged_data['ratio_height_diam']     = np.where(merged_data['Diameter']!=0,merged_data['Height']/merged_data['Diameter'],np.NAN)
aggregation_columns                  = ['Height','Diameter','MCA1','PCA1','ratio_height_diam']

merged_data                          = updated_df(merged_data,'Area_Code','mean',aggregation_columns)
merged_data                          = updated_df(merged_data,'Area_Code','std',aggregation_columns)
merged_data                          = updated_df(merged_data,'Area_Code','min',aggregation_columns)
merged_data                          = updated_df(merged_data,'Area_Code','max',aggregation_columns)
merged_data                          = updated_df(merged_data,'Area_Code','median',aggregation_columns)


merged_data                          = updated_df(merged_data,'Locality_Code','mean',aggregation_columns)
merged_data                          = updated_df(merged_data,'Locality_Code','std',aggregation_columns)
merged_data                          = updated_df(merged_data,'Locality_Code','min',aggregation_columns)
merged_data                          = updated_df(merged_data,'Locality_Code','max',aggregation_columns)
merged_data                          = updated_df(merged_data,'Locality_Code','median',aggregation_columns)

merged_data                          = updated_df(merged_data,'Region_Code','mean',aggregation_columns)
merged_data                          = updated_df(merged_data,'Region_Code','std',aggregation_columns)
merged_data                          = updated_df(merged_data,'Region_Code','min',aggregation_columns)
merged_data                          = updated_df(merged_data,'Region_Code','max',aggregation_columns)
merged_data                          = updated_df(merged_data,'Region_Code','median',aggregation_columns)

merged_data                          = updated_df(merged_data,'Species','mean',aggregation_columns)
merged_data                          = updated_df(merged_data,'Species','std',aggregation_columns)
merged_data                          = updated_df(merged_data,'Species','min',aggregation_columns)
merged_data                          = updated_df(merged_data,'Species','max',aggregation_columns)
merged_data                          = updated_df(merged_data,'Species','median',aggregation_columns)

merged_data                          = updated_df(merged_data,'Area_Code','nunique',['Species'])
merged_data                          = updated_df(merged_data,'Locality_Code','nunique',['Species'])
merged_data                          = updated_df(merged_data,'Region_Code','nunique',['Species'])

merged_data                          = updated_df(merged_data,'Area_Code','nunique',['Locality_Code'])
merged_data                          = updated_df(merged_data,'Region_Code','nunique',['Locality_Code'])



testcount  = len(testdf)
count      = len(merged_data)-testcount

traindf = merged_data[:count]
testdf  = merged_data[count:]

for cols in categorical_cols:
    traindf[cols] = traindf[cols].astype(str)
    testdf[cols]  = testdf[cols].astype(str)


train            = traindf.values
test             = testdf.values


cate_features_index = np.where(traindf.dtypes == object)[0]


oof_pred               = np.zeros((len(train),8 ))
y_pred_1               = np.zeros((len(test),8 ))

n_splits               = 50
kf                     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=202020)

for fold, (tr_ind, val_ind) in enumerate(kf.split(train, train_y)):
    
    X_train, X_val     = train[tr_ind], train[val_ind]
    y_train, y_val     = train_y[tr_ind], train_y[val_ind]
    
    
    
    model1 = CatBoostClassifier(n_estimators=2000,random_state=202020,verbose=False)
    model1.fit(X_train,y_train,cat_features = cate_features_index,eval_set=(X_val,y_val))
    val_pred1 = model1.predict_proba(X_val)
    y_pred_1  += model1.predict_proba(test) / (n_splits)
    
    
    print('validation loglos -',fold+1,': ',log_loss(y_val,val_pred1))
    
    oof_pred[val_ind]  = val_pred1
    print('\n')
    
print('OOF logloss:- ',(log_loss(train_y,oof_pred)))



sample_submission   = pd.read_csv(r'C:\Users\Salim\Downloads\Participants_Data\Sample_Submission.csv')
columns_name        = sample_submission.columns.tolist()
finaldf             = pd.DataFrame(y_pred_1,columns=columns_name)



for cols in finaldf.columns:
    finaldf[cols] = finaldf[cols].apply(lambda z :1.0 if z >0.85 else z)


finaldf.to_csv(r"C:\Users\Salim\Downloads\Participants_Data\singlemodel_50folds.csv",index=False)

