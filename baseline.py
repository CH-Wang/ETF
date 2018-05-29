import numpy as np 
import pandas as pd
from data_loader import score_cal

## Persistence Model
class PersistenceModel():

    def fit(self, data, target):
        return True

    def predict(self, data):
        output = []
        for i in data:
            output.append([i[-1] for j in range(5)])
        return output
    
if __name__ == '__main__':
    train_df = pd.read_csv('../data/train.csv', sep=',',header=None)
    test_df = pd.read_csv('../data/test.csv', sep=',',header=None)

    train_input = train_df.iloc[1:,1:-5].values
    train_target = train_df.iloc[1:,-5:].values

    test_input = test_df.iloc[1:,1:-5].values
    test_target = test_df.iloc[1:,-5:].values


    baseline = PersistenceModel()
    baseline.fit(train_input,train_target)

    test_output = baseline.predict(test_input)


    score = score_cal(test_input, test_target, test_output)

    print(score)