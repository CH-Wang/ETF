import pandas as pd
import numpy as np
from SVM import SVM
from baseline import PersistenceModel
from data_loader import score_cal
from LSTM import LSTM



train_df = pd.read_csv('../data/train.csv', sep=',',header=None)
test_df = pd.read_csv('../data/test.csv', sep=',',header=None)

train_input = train_df.iloc[1:,1:-5].values
train_target = train_df.iloc[1:,-5:].values

test_input = test_df.iloc[1:,1:-5].values
test_target = test_df.iloc[1:,-5:].values


svm = SVM()
svm.fit(train_input,train_target)
svm.save()
test_output = svm.predict(test_input)
svm_score = score_cal(test_input, test_target, test_output)
print('svm score:', svm_score)


baseline = PersistenceModel()
baseline.fit(train_input,train_target)
test_output = baseline.predict(test_input)
baseline_score = score_cal(test_input, test_target, test_output)
print('baseline score:', baseline_score)



lstm = LSTM()
# lstm.fit('../data/train.csv', n_epoch=15)
# lstm.save()
lstm.load()
lstm_score = lstm.score('../data/train.csv')
print ('lstm score:', lstm_score)