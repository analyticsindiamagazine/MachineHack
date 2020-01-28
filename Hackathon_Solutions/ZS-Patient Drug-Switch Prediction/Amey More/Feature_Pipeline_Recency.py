import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os

time_var = 'event_time'
payment_var = 'patient_payment'
id_var = 'patient_id'
y_var = 'outcome_flag'

def rec_feature_creator(new_df, df):
    cat_col = ['event_name', 'specialty', 'plan_type']
    for cat in cat_col:
        rec_df = df.groupby(['patient_id', cat])[time_var].min().unstack(cat)
        rec_df = rec_df.add_prefix('recency_')
        new_df = new_df.merge(rec_df, left_index=True, right_index=True, how='left')
    return new_df

def fitness_calculation(data):
    if ((data['sd_0'] == 0 ) and (data['sd_1'] == 0)) and (((data['avg_0'] == 0) and (data['avg_1'] != 0)) or ((data['avg_0'] != 0) and (data['avg_1'] == 0))):
        return 9999999999
    elif (((data['sd_0'] == 0 ) and (data['sd_1'] != 0)) or ((data['sd_0'] != 0) and (data['sd_1'] == 0))) and (data['avg_0'] == data['avg_1']):
        return 1
    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and (data['avg_0'] != 0):
        return ((data['avg_1']/data['sd_1'])/(data['avg_0']/data['sd_0']))
    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and ((data['avg_0'] == 0) and (data['avg_1'] != 0)):
        return 9999999999
    else:
        return 1

def main():
    print("Reading datasets...")
    train = pd.read_csv('./train_data.csv')
    test = pd.read_csv('./test_data.csv')
    train_lables = pd.read_csv('./train_labels.csv')
    sample_sub = pd.read_csv('./Sample Submission.csv')

    print("Concatenate train and test into new dataframe...")
    df = pd.concat([train,test],axis=0).reset_index(drop=True)
    del train
    del test
    df.drop('patient_payment',axis=1,inplace=True)

    print("Creating final_df with all patient_ids as single column...")
    patient_ids = train_lables.patient_id.tolist() + sample_sub.patient_id.tolist()
    del sample_sub
    final_df = pd.DataFrame({'patient_id': patient_ids})
    final_df.set_index('patient_id', inplace=True)

    cat_col = ['event_name', 'specialty', 'plan_type']
    for cat in cat_col:
        df[cat] = df[cat].apply(lambda x: cat + '__' +x)

    print("Calling rec_feature_creator method...")
    rec_feats_df = rec_feature_creator(final_df, df)

    gc.collect()

    train_lables.set_index('patient_id',inplace=True)

    print("Joining training target variables to newly created dataframe...")
    rec_feats_df = rec_feats_df.merge(train_lables, left_index=True, right_index=True, how='left')

    print("Calculating Fitness Values...")
    fitness_values_df = rec_feats_df[rec_feats_df.outcome_flag.notnull()].groupby('outcome_flag').describe().stack(0).unstack('outcome_flag')[['mean', 'std']]
    fitness_values_df.columns = ['_'.join([str(i) for i in x]) for x in fitness_values_df.columns.ravel()]
    fitness_values_df.rename({'mean_0.0': 'avg_0', 'mean_1.0': 'avg_1', 'std_0.0': 'sd_0', 'std_1.0': 'sd_1'},axis=1,inplace=True)
    fitness_values_df['fitness_value'] = fitness_values_df.apply(fitness_calculation, axis=1)

    rec_feats_df.drop('outcome_flag',axis=1,inplace=True)

    print("Exporting files recency_fitness_values.csv & recency_feats_df.csv")
    fitness_values_df.to_csv('Fitness_Score_recency.csv',index_label='feature_name')
    rec_feats_df.to_csv('recency_feats_df.csv')

if __name__ == "__main__":
    main()
