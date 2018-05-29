import numpy as np 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import pandas as pd
from data_loader import denormalize

def score_cal(denorm_input, denorm_target, denorm_output):
    score_list = []
    last_price = denorm_input[-1].round(2)
    for i in range(5):
        score = 0
        target_price = denorm_target[i].round(2)
        ouput_price = denorm_output[i].round(2)
        diff_target = target_price - last_price
        diff_ouput = ouput_price - last_price
        last_price = target_price
        if (diff_target*diff_ouput > 0 or diff_target == diff_ouput):
            score += 0.5
        score += (target_price - abs(target_price - ouput_price))/target_price*0.5
        score_list.append(score)
    weighted_score = map(lambda x,y:x*y,score_list,[0.1,0.15,0.2,0.25,0.3])
    return sum(weighted_score)



train_df = pd.read_csv('../data/train.csv', sep=',',header=None)
test_df = pd.read_csv('../data/test.csv', sep=',',header=None)

train_input = train_df.iloc[1:,1:-5].values
train_target = train_df.iloc[1:,-5:].values

test_input = test_df.iloc[1:,1:-5].values
test_target = test_df.iloc[1:,-5:].values


MOR = MultiOutputRegressor(SVR(kernel='rbf', C=1e3, gamma=0.1))
MOR.fit(train_input,train_target)

test_output = MOR.predict(test_input)



df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'Big5')
df = pd.DataFrame(df) 

denorm_input = denormalize(df, test_input)
denorm_target = denormalize(df, test_target)
denorm_output = denormalize(df, test_output)

score = 0
for i, d_input in enumerate(denorm_input):
    d_target = denorm_target[i]
    d_output = denorm_output[i]
    score += score_cal(d_input, d_target, d_output)

avg_score = score/ len(denorm_input)

print(avg_score)
