import pandas as pd
import numpy as np
import torch
from SVM import SVM
from baseline import PersistenceModel
from dataLoader import scoreCal
from LSTM import LSTM
from ANN import ANN
from ANN import ETFDataset


train_df = pd.read_csv('../data/train.csv', sep=',',header=None)
test_df = pd.read_csv('../data/test.csv', sep=',',header=None)

train_input = train_df.iloc[1:,1:-5].values
train_target = train_df.iloc[1:,-5:].values

test_input = test_df.iloc[1:,1:-5].values
test_target = test_df.iloc[1:,-5:].values


## Baseline
baseline = PersistenceModel()
baseline.fit(train_input,train_target)
test_output = baseline.predict(test_input)
baseline_score = scoreCal(test_input, test_target, test_output)
print('baseline score:', baseline_score)

## SVM
svm = SVM()
svm.fit(train_input,train_target)
svm.save()
test_output = svm.predict(test_input)
svm_score = scoreCal(test_input, test_target, test_output)
print('svm score:', svm_score)

## ANN
ann = ANN()
# ann.fit('../data/train.csv', n_epoch=40)
# ann.save()
ann.load()
ann_score = ann.score('../data/test.csv')
print ('ann score:', ann_score)



## LSTM
lstm = LSTM()
# lstm.fit('../data/train.csv', n_epoch=70)
# lstm.save()
lstm.load()
lstm_score = lstm.score('../data/test.csv')
print ('lstm score:', lstm_score)
