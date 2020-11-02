#!/usr/bin/env python
# coding: utf-8

# Import Important packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import StratifiedKFold,KFold
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import os

path =  "../input/mh-github-bug/Embold_Participant's_Dataset/"

TRAIN_PATH = path+'embold_train.json'
TRAIN_PATH1 = path+'embold_train_extra.json'
TEST_PATH = path+'embold_test.json'
SAMPLE_SUB_PATH = path+'sample submission.csv'
train = pd.read_json(TRAIN_PATH)
train1 = pd.read_json(TRAIN_PATH1)
test = pd.read_json(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


train = pd.concat([train,train1])
train.reset_index(drop=True,inplace=True)
train.shape,train1.shape,test.shape,sample_sub.shape


df_train=train.copy()
df_test=test.copy()


print('there are {} rows and {} columns in the train'.format(df_train.shape[0],df_train.shape[1]))
print('there are {} rows and {} columns in the test'.format(df_test.shape[0],df_test.shape[1]))


df_train.head(3)



def quick_encode(df,maxlen=100):
    
    values = df[['title','body']].values.tolist()
    tokens=tokenizer.batch_encode_plus(values,max_length=maxlen,pad_to_max_length=True)
    
    return np.array(tokens['input_ids'])

def create_dist_dataset(X, y,val,batch_size= BATCH_SIZE):   
    
    dataset = tf.data.Dataset.from_tensor_slices((X,y)).shuffle(len(X))
          
    if not val:
        dataset = dataset.repeat().batch(batch_size).prefetch(AUTO)
    else:
        dataset = dataset.batch(batch_size).prefetch(AUTO)    
    
    return dataset

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_test))
    .batch(BATCH_SIZE)
)


# Model

def build_model(transformer,max_len):
    
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    sequence_output = transformer(input_ids)[0]
    cls_token = sequence_output[:, 0, :]

    out = Dense(3, activation='softmax')(cls_token)

    # It's time to build and compile the model
    model = Model(inputs=input_ids, outputs=out)
    model.compile(
        Adam(lr=1e-5), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


# Kfold CV

def run_kf(MODEL,EPOCHS,MAX_LEN,model_nm)

    x_train = quick_encode(df_train)
    x_test = quick_encode(df_test)
    y_train = df_train.label.values

    # Our batch size will depend on number of replic
    BATCH_SIZE= 16 
    AUTO = tf.data.experimental.AUTOTUNE
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pred_test=np.zeros((df_test.shape[0],3))
    skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=777)
    val_score=[]
    history=[]


    for fold,(train_ind,valid_ind) in enumerate(skf.split(x_train,y_train)):

        if fold < 4:

            print("fold",fold+1)


            tf.tpu.experimental.initialize_tpu_system(tpu)

            train_data = create_dist_dataset(x_train[train_ind],y_train[train_ind],val=False)
            valid_data = create_dist_dataset(x_train[valid_ind],y_train[valid_ind],val=True)

            Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"roberta_base.h5", monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='min')

            with strategy.scope():
                transformer_layer = TFAutoModel.from_pretrained(MODEL)
                model = build_model(transformer_layer, max_len=MAX_LEN)



            n_steps = len(train_ind)
            print("training model {} ".format(fold+1))

            train_history = model.fit(
            train_data,
            steps_per_epoch=n_steps,
            validation_data=valid_data,
            epochs=EPOCHS,callbacks=[Checkpoint],verbose=1)

            print("Loading model...")
            model.load_weights(f"roberta_base.h5")



            print("fold {} validation accuracy {}".format(fold+1,np.mean(train_history.history['val_accuracy'])))
            print("fold {} validation loss {}".format(fold+1,np.mean(train_history.history['val_loss'])))

            val_score.append(train_history.history['val_accuracy'])
            history.append(train_history)

            val_score.append(np.mean(train_history.history['val_accuracy']))

            print('predict on test....')
            preds=model.predict(test_dataset,verbose=1)

            pred_test+=preds



    print("Mean Validation accuracy : ",np.mean(val_score))


    # Evaluation

    plt.figure(figsize=(15,10))

    for i,hist in enumerate(history):

        plt.subplot(2,2,i+1)
        plt.plot(np.arange(EPOCHS),hist.history['accuracy'],label='train accu')
        plt.plot(np.arange(EPOCHS),hist.history['val_accuracy'],label='validation acc')
        plt.gca().title.set_text(f'Fold {i+1} accuracy curve')
        plt.legend()


    plt.figure(figsize=(15,10))

    for i,hist in enumerate(history):

        plt.subplot(2,2,i+1)
        plt.plot(np.arange(EPOCHS),hist.history['loss'],label='train loss')
        plt.plot(np.arange(EPOCHS),hist.history['val_loss'],label='validation loss')
        plt.gca().title.set_text(f'Fold {i+1} loss curve')
        plt.legend()


    # Submission


    sub = pd.DataFrame(pred_test,columns = '0 1 2'.split())
    sub.to_csv(model_nm+'pred.csv', index=False)

    submission = sample_sub
    submission['label'] = np.argmax(pred_test,axis=1)
    submission.head()

    submission.to_csv(model_nm+'submission.csv',index=False)

## Run xlm-roberta-large-kfold 

MODEL = 'jplu/tf-xlm-roberta-large'
EPOCHS = 7
MAX_LEN = 128
model_nm = 'xlm roberta large kfold'

run_kf(MODEL=MODEL,EPOCHS = EPOCHS,MAX_LEN=MAX_LEN,model_nm=model_nm)

## Run 

MODEL = 'roberta-base'
EPOCHS = 6
MAX_LEN = 128
model_nm = 'roberta base kfold'
run_kf(MODEL=MODEL,EPOCHS = EPOCHS,MAX_LEN=MAX_LEN,model_nm=model_nm)

