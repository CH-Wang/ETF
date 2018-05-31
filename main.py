import pandas as pd
import numpy as np
import torch
from SVM import SVM
from baseline import PersistenceModel
from dataLoader import scoreCal
from LSTM import LSTM
from ANN import ANN
from ANN import ETFDataset

if __name__ == '__main__':

    ETFcode = [50,51,52,53,54,55,56,57,58,59,6201,6203,6204,6208,690,692,701,713]

    for code in [50]:

        trainPath = '../data/train'+str(code)+'.csv'
        testPath = '../data/test'+str(code)+'.csv'

        train_df = pd.read_csv(trainPath, sep=',',header=None)
        test_df = pd.read_csv(testPath, sep=',',header=None)

        train_input = train_df.iloc[1:,1:-5].values
        train_target = train_df.iloc[1:,-5:].values

        test_input = test_df.iloc[1:,1:-5].values
        test_target = test_df.iloc[1:,-5:].values


        ## Baseline1
        baseline = PersistenceModel()
        baseline.fit(train_input,train_target)
        test_output = baseline.predict(test_input)
        baseline_score = scoreCal(test_input, test_target, test_output)
        print('baseline1 score:', baseline_score)

        ## Baseline2
        baseline_score = scoreCal(test_input, test_target, test_output, variation=[1,1,1,1,1])
        print('baseline2 score:', baseline_score)

        ## SVM
        svm = SVM()
        # svm.fit(train_input,train_target)
        # svm.save()
        svm.load()
        test_output = svm.predict(test_input)
        svm_score = scoreCal(test_input, test_target, test_output)
        print('svm score:', svm_score)
        print('svm abs score:', scoreCal(test_input, test_target, test_output, count_variation=False))

        ## ANN
        ann = ANN()
        # ann.fit('../data/train.csv', n_epoch=40)
        # ann.save()
        ann.load()
        ann_score = ann.score(testPath)
        print('ann score:', ann_score)
        print('ann abs score:', ann.score(testPath, count_variation=False))

        ## LSTM
        lstm = LSTM()
        # lstm.fit('../data/train.csv', n_epoch=70)
        # lstm.save()
        lstm.load()
        lstm_score = lstm.score(testPath)
        print ('lstm score:', lstm_score)
        print('lstm abs score:', lstm.score(testPath, count_variation=False))