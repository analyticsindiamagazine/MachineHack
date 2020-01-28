import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

time_var = 'event_time'
payment_var = 'patient_payment'
id_var = 'patient_id'
y_var = 'outcome_flag'


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

def freq_feature_creator(new_df, df, cat):
    df['frequency__'+cat] = 'frequency_' + df['bucketed_time'] + '_' + cat + '__' + df[cat]
    temp_df = df.groupby([id_var, 'frequency__'+cat]).agg({'frequency__'+cat: 'count'}).unstack('frequency__'+cat)
    temp_df.columns=temp_df.columns.droplevel(0)
    new_df = new_df.merge(temp_df, left_index=True, right_index=True, how='left')
    return new_df

def fitness_value_creator(feats_df, train_lables):
    feats_df.fillna(0,inplace=True)
    feats_df = feats_df.merge(train_lables, left_index=True, right_index=True, how='left')
    fitness_values_df = feats_df.groupby('outcome_flag').describe().stack(0).unstack('outcome_flag')[['mean', 'std']]
    del feats_df
    gc.collect()
    fitness_values_df.columns = ['_'.join([str(i) for i in x]) for x in fitness_values_df.columns.ravel()]
    fitness_values_df.rename(columns={'mean_0': 'avg_0', 'mean_1': 'avg_1', 'std_0': 'sd_0', 'std_1': 'sd_1'},inplace=True)
    fitness_values_df['fitness_value'] = fitness_values_df.apply(fitness_calculation, axis=1)
    return fitness_values_df


def main():
    print("Reading datasets...\n")
    train = pd.read_csv('./train_data.csv')
    test = pd.read_csv('./test_data.csv')
    train_lables = pd.read_csv('./train_labels.csv')

    sample_sub = pd.read_csv('./Sample Submission.csv')

    df = pd.concat([train,test],axis=0).reset_index(drop=True)
    del train
    del test
    df.drop('patient_payment',axis=1,inplace=True)

    patient_ids = train_lables.patient_id.tolist() + sample_sub.patient_id.tolist()
    del sample_sub
    final_df = pd.DataFrame({'patient_id': patient_ids})
    final_df.set_index('patient_id', inplace=True)

    train_lables.set_index('patient_id',inplace=True)

    print("Creating bins for binning...")
    bins = [0] + [i*30 for i in range(1,37)]
    df['bucketed_time'] = pd.cut(x=df['event_time'], bins=bins, include_lowest=True, labels=bins[1:]).astype('str')

    gc.collect()

    freq_fitness_values_df = pd.DataFrame()

    cat = 'event_name'
    print("Creating features for {}...\n".format(cat))
    event_freq_feat_df = freq_feature_creator(final_df, df, cat)
    event_freq_feat_df.to_csv('event_freq_feat_df.csv')
    feats_with_target = event_freq_feat_df[:train_lables.shape[0]]
    del event_freq_feat_df
    df_list = np.array_split(feats_with_target, (feats_with_target.shape[0] / 600), axis=1)
    gc.collect()
    for df_iterator in df_list:
        freq_fitness_values_df = pd.concat([freq_fitness_values_df, fitness_value_creator(df_iterator, train_lables)])

    cat = 'specialty'
    print("Creating features for {}...\n".format(cat))
    spec_freq_feat_df = freq_feature_creator(final_df, df, cat)
    spec_freq_feat_df.to_csv('spec_freq_feat_df.csv')
    feats_with_target = spec_freq_feat_df[:train_lables.shape[0]]
    del spec_freq_feat_df
    df_list = np.array_split(feats_with_target, (feats_with_target.shape[0] / 600), axis=1)
    gc.collect()
    for df_iterator in df_list:
        freq_fitness_values_df = pd.concat([freq_fitness_values_df, fitness_value_creator(df_iterator, train_lables)])

    cat = 'plan_type'
    print("Creating features for {}...\n".format(cat))
    plan_freq_feat_df = freq_feature_creator(final_df, df, cat)
    plan_freq_feat_df.to_csv('plan_freq_feat_df.csv')
    feats_with_target = plan_freq_feat_df[:train_lables.shape[0]]
    del plan_freq_feat_df
    df_list = np.array_split(feats_with_target, (feats_with_target.shape[0] / 600), axis=1)
    gc.collect()
    for df_iterator in df_list:
        freq_fitness_values_df = pd.concat([freq_fitness_values_df, fitness_value_creator(df_iterator, train_lables)])

    freq_fitness_values_df.to_csv('Fitness_Score_frequency.csv',index_label='feature_name')

if __name__ == "__main__":
    main()
