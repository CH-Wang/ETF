import numpy as np 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import pandas as pd
from data_loader import score_cal



train_df = pd.read_csv('../data/train.csv', sep=',',header=None)
test_df = pd.read_csv('../data/test.csv', sep=',',header=None)

train_input = train_df.iloc[1:,1:-5].values
train_target = train_df.iloc[1:,-5:].values

test_input = test_df.iloc[1:,1:-5].values
test_target = test_df.iloc[1:,-5:].values


MOR = MultiOutputRegressor(SVR(kernel='rbf', C=1e3, gamma=0.1))
MOR.fit(train_input,train_target)

test_output = MOR.predict(test_input)

score = score_cal(test_input, test_target, test_output)

print(score)
