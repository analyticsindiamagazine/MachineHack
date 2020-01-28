import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

time_var = 'event_time'
payment_var = 'patient_payment'
id_var = 'patient_id'
y_var = 'outcome_flag'

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    sparse_flag = False
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def pay_feats_creator(new_df, df):
    cat_list = ['event_name', 'specialty']
    for cat in cat_list:
        payment_df = df.groupby(['patient_id', cat]).agg({payment_var: ['min','max','mean','sum']}).unstack(cat)
        payment_df.columns = ["_".join(x) for x in payment_df.columns.ravel()]
        payment_df.reset_index(inplace=True)
        new_df = pd.merge(new_df, payment_df, on='patient_id', how='left')
    return new_df

def overall_freq_feats_creator(new_df, df):
    cat_list = ['event_name']
    for cat in cat_list:
        payment_df = df.groupby(['patient_id', cat]).agg({time_var: 'count'}).unstack(cat)
        payment_df.columns = ["_".join(x) for x in payment_df.columns.ravel()]
        payment_df.reset_index(inplace=True)
        new_df = pd.merge(new_df, payment_df, on='patient_id', how='left')
    return new_df

def main():
    print("Reading datasets...\n")
    train = pd.read_csv('./train_data.csv')
    test = pd.read_csv('./test_data.csv')
    train_lables = pd.read_csv('./train_labels.csv')

    print("Concatenate train and test into new dataframe...\n")
    df = pd.concat([train,test],axis=0).reset_index(drop=True)
    del train
    del test

    recency_df = pd.read_csv('recency_feats_df.csv')
    print("Reducing Memory Usage of Recency Features Dataset...\n")
    recency_df = reduce_mem_usage(recency_df)

    gc.collect()

    print("Creating aggregate features from patient_payment column...\n")
    recency_df = pay_feats_creator(recency_df, df)

    print("Creating overall frequency feature over event_name column...\n")
    recency_df = overall_freq_feats_creator(recency_df, df)


    print("Creating aggregate features from patient_payment column over patient_id...\n")
    patient_payment_df = df.groupby(['patient_id'],as_index=False).agg({payment_var: ['min','max','mean','sum']})
    patient_payment_df.columns = ["_".join(x) for x in patient_payment_df.columns.ravel()]
    patient_payment_df.rename({"patient_id_": 'patient_id'}, axis=1, inplace=True)
    recency_df = pd.merge(recency_df, patient_payment_df, on='patient_id', how='left')

    print("Creating count column for patient_id...\n")
    patient_events_count = df.groupby(['patient_id'],as_index=False).agg({'event_name': 'count'}).rename({'event_name': 'patient_event_count'}, axis=1)
    recency_df = pd.merge(recency_df, patient_events_count, on='patient_id', how='left')


    recency_df = pd.merge(recency_df, train_lables, on='patient_id', how='left')

    print("Dropping columns present in train but not in test set...\n")
    drop_cols = ['spec_111', 'spec_123', 'spec_139', 'spec_141', 'spec_143',
           'spec_158', 'spec_165', 'spec_170', 'spec_172', 'spec_174',
           'spec_181', 'spec_183', 'spec_188', 'spec_194', 'spec_195',
           'spec_197', 'spec_200', 'spec_201', 'spec_202', 'spec_205',
           'spec_209', 'spec_211', 'spec_213', 'spec_214', 'spec_215',
           'spec_216', 'spec_222', 'spec_225', 'spec_227', 'spec_228',
           'spec_230', 'spec_231', 'spec_232', 'spec_238', 'spec_239',
           'spec_241']
    drop_cols = [col for col in recency_df.columns if any(d_c in col for d_c in drop_cols)]
    recency_df.drop(drop_cols, axis=1,inplace=True)

    # In[ ]:

    print("Exporting final dataframe with {} features".format(recency_df.shape[1] - 1))
    recency_df.to_csv('final_df.csv',index=False)

if __name__ == "__main__":
    main()
