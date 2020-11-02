#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

from IPython.display import HTML
import base64

# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "roberta_large.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# ## Helper Functions

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])


def build_model(transformer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]

    out = Dense(3, activation='softmax')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# ## Load text data into memory

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

from sklearn.model_selection import train_test_split

train['text'] = train['title'] + " " + train['body']
test['text'] = test['title'] + " " + test['body']

train,valid = train_test_split(train[['text','label']],test_size=.005,random_state=2020)


train.rename(columns= {'label':'toxic','text':'comment_text'},inplace=True)
valid.rename(columns= {'label':'toxic','text':'comment_text'},inplace=True)
test.rename(columns= {'text':'comment_text'},inplace=True)


def run_model(EPOCHS,MAX_LEN,MODEL,model_nm):
    BATCH_SIZE = 16 
    # First load the real tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    x_train = regular_encode(train.comment_text.values, tokenizer, maxlen=MAX_LEN)
    x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)
    x_test = regular_encode(test.comment_text.values, tokenizer, maxlen=MAX_LEN)

    from keras.utils import np_utils
    y_train = (train.toxic)
    y_valid = (valid.toxic)


    # ## Build datasets objects

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )

    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(BATCH_SIZE)
    )


    # ## Load model into the TPU


    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model(transformer_layer, max_len=MAX_LEN)
    model.summary()


    # ## Train Model

    n_steps = x_train.shape[0] 
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs= EPOCHS
    )

    pred_val = model.predict(valid_dataset, verbose=1)

    from sklearn.metrics import accuracy_score
    print('accracy val',accuracy_score(np.argmax(pred_val,1),valid.toxic))


    # ## Submission

    sample_sub.head()

    pred_test = model.predict(test_dataset, verbose=1)
    pd.value_counts(np.argmax(pred_test,1))

    sub = pd.DataFrame(pred_test,columns = '0 1 2'.split())
    sub.to_csv(model_nm+'pred.csv', index=False)

    sample_sub['label'] = np.argmax(pred_test,1)

    create_download_link(sample_sub,filename = model_nm+'submission.csv')

# Run Roberta large
EPOCHS = 3
MAX_LEN = 128
MODEL = 'roberta-large'
model_nm= 'Roberta large'

run_model(EPOCHS=EPOCHS,MAX_LEN=MAX_LEN,MODEL=MODEL,model_nm=model_nm)

# Run XLM Roberta large
EPOCHS = 4
MAX_LEN = 128
MODEL = 'jplu/tf-xlm-roberta-large'
model_nm= 'XLM-Roberta large'

run_model(EPOCHS=EPOCHS,MAX_LEN=MAX_LEN,MODEL=MODEL,model_nm=model_nm)

