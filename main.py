import pandas as pd
import numpy as np
import torch
from dataLoader import scoreCal
from dataLoader import codeDenormalize
from SVM import SVM
from baseline import PersistenceModel
from LSTM import LSTM
from ANN import ANN
from ANN import ETFDataset

if __name__ == '__main__':

    ETFcode = [50,51,52,53,54,55,56,57,58,59,6201,6203,6204,6208,690,692,701,713]

    for code in [50]:

        trainPath = '../data/train'+str(code)+'.csv'
        testPath = '../data/test'+str(code)+'.csv'

        train_df = pd.read_csv(trainPath)
        test_df = pd.read_csv(testPath)

        train_input = train_df.iloc[:,1:-5].values
        train_target = train_df.iloc[:,-5:].values
        test_input = test_df.iloc[:,1:-5].values
        test_target = test_df.iloc[:,-5:].values
        
        denorm_test_input = codeDenormalize(test_input, code=code)
        denorm_test_target = codeDenormalize(test_target, code=code)       
        
        last_data = test_df.iloc[-1, 1:].values

        ## Baseline1
        baseline = PersistenceModel()
        baseline.fit(train_input,train_target)
        test_output = np.array(baseline.predict(test_input))
        ## denormalize
        test_output = codeDenormalize(test_output, code=code)
        test_data = denorm_test_input
        test_target = denorm_test_target     
        ## score
        baseline_score = scoreCal(test_input, test_target, test_output)
        print('baseline1 score:', baseline_score)
        ## predict
        predict = np.array(baseline.predict([last_data]))
        predict = codeDenormalize(predict, code=code)    
        print('prediction:', predict, '\n\n')

        ## Baseline2
        baseline_score = scoreCal(test_input, test_target, test_output, variation=[1,1,1,1,1])
        print('baseline2 score:', baseline_score, '\n\n')

        ## SVM
        svm = SVM()
        svm.fit(train_input,train_target)
        svm.save()
        # svm.load()
        test_output = svm.predict(test_input)
        ## denormalize
        test_output = codeDenormalize(test_output, code=code)
        test_data = denorm_test_input
        test_target = denorm_test_target
        ## score
        svm_score = scoreCal(test_input, test_target, test_output)
        print('svm score:', svm_score)
        print('svm abs score:', scoreCal(test_input, test_target, test_output, count_variation=False))       
        ## predict
        predict = svm.predict([last_data[-15:]])
        predict = codeDenormalize(predict, code=code)    
        print('prediction:', predict, '\n\n')

        ## ANN
        ann = ANN()
        # ann.fit(trainPath, n_epoch=40)
        # ann.save()
        ann.load()
        test_output = ann.predictTestSet(testPath)

        ## denormalize
        test_output = codeDenormalize(test_output, code=code)
        test_data = codeDenormalize(test_df.iloc[:,1:16].values, code=code)
        test_target = codeDenormalize(test_df.iloc[:,16:].values, code=code)

        ## score
        ann_score = scoreCal(test_data,test_target,test_output)
        print ('ann score:', ann_score)
        print('ann abs score:', scoreCal(test_data,test_target,test_output, count_variation=False))
        
        ## predict
        predict = ann.predict(last_data[-15:])
        predict = codeDenormalize(predict, code=code)        
        print('prediction: \n', predict, '\n\n')

        ## LSTM
        future = 4
        lstm = LSTM()
        # lstm.fit(trainPath, n_epoch=70)
        # lstm.save()
        lstm.load()
        test_output = lstm.predictTestSet(testPath, future=future)

        ## denormalize
        test_outtput = codeDenormalize(test_output[:, (-future-1):], code=code)
        test_data = codeDenormalize(test_df.iloc[:,1:(-future-1)].values, code=code)
        test_target = codeDenormalize(test_df.iloc[:,(-future-1):].values, code=code)        
        
        ## score
        lstm_score = scoreCal(test_data,test_target,test_output)
        print ('lstm score:', lstm_score)
        print('lstm abs score:', scoreCal(test_data,test_target,test_output, count_variation=False))
        
        ## predict
        predict = lstm.predict(last_data, future=4)
        predict = codeDenormalize(predict, code=code)    
        print('prediction: \n', predict, '\n\n')