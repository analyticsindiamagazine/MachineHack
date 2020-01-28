import pandas as pd
import numpy as np
import sys
from custom_estimator import Estimator
from encoding import FreqeuncyEncoding
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os
import glob
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore")

def main():
    data_directory = './'
    print("Reading Dataset...\n")
    df=pd.read_csv(data_directory+'final_df.csv')

    print("DataFrame shape: {}".format(df.shape))

    train = df.iloc[:16683]
    test = df.iloc[16683:]

    cat_cols=None
    target=['outcome_flag']
    drop_cols=['patient_id']

    print("Train Shape: {}".format(train.shape))

    use_cols=df.columns[~df.columns.isin(drop_cols+target)]

    print("Initializing LGBMClassifier...\n")
    mod = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.4,
                   importance_type='split', learning_rate=0.01, max_depth=-1,
                   metric='None', min_child_samples=20, min_child_weight=20.0,
                   min_split_gain=0.0, n_estimators=5000, n_jobs=-1, num_leaves=48,
                   objective='binary', random_state=None, reg_alpha=0.0,
                   reg_lambda=0.0, silent=True, subsample=0.9,
                   subsample_for_bin=200000, subsample_freq=5)

    print("Initializing Estimator...\n")
    est_lgb=Estimator(model=mod,n_jobs=-1,early_stopping_rounds=200)

    print("Training...")
    oof_preds = est_lgb.fit_transform(train[use_cols].values,train['outcome_flag'].values)
    
    print("Avg. CV Score: {}".format(est_lgb.avg_cv_score))

    preds = est_lgb.transform(test[use_cols].values)

    print("Exporting prediction probabilities to final_sub...\n")
    pd.DataFrame({'patient_id':test['patient_id'],'outcome_flag':preds}).to_csv('final_sub.csv',index=False)

    print("Exporting feature importances to feature_importance.csv...\n")
    est_lgb.feature_importance_df(use_cols).to_csv('feature_importance.csv',index=False)


    # ### thresholding for hard classes

    print("Calculating threshold for class prediction...\n")
    thresholds = np.arange(0.01,0.99,0.01)

    from sklearn.metrics import roc_auc_score

    max_val = np.argmax([roc_auc_score(train['outcome_flag'],(oof_preds>x).astype('int')) for x in thresholds])

    thresholds[max_val]

    roc_auc_score(train['outcome_flag'],(oof_preds>thresholds[max_val]).astype('int'))

    hard_preds = (preds>thresholds[max_val]).astype('int')

    print("Exporting hard class predictions to Best Score.xlsx")
    pd.DataFrame({'patient_id':test['patient_id'],'outcome_flag':hard_preds}).to_excel('Best Score.xlsx',index=False, columns=['patient_id', 'outcome_flag'])

if __name__ == "__main__":
    main()
