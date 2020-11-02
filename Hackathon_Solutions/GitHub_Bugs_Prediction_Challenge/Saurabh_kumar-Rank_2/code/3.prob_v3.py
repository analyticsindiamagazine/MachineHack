#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

sub1 = pd.read_csv('xlm roberta large kfoldpred.csv')
sub2 = pd.read_csv('Roberta largepred.csv')
sub3 = pd.read_csv('XLM-Roberta largesubmission.csv')
sub4 = pd.read_csv('roberta base kfoldpred.csv')

sub_all = sub1.values*.2 + sub2.values*.2 + sub3.values*.3 + sub4.values*.3
pd.value_counts(np.argmax(sub_all,1))

sub = pd.DataFrame(np.argmax(sub_all,1),columns = ['label'])
sub.head(2)

sub.to_csv('prob4.csv',index=False)

