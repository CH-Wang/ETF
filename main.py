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
        baseline_predict = np.array(baseline.predict([last_data]))
        baseline_predict = codeDenormalize(baseline_predict, code=code)    
        print('baseline prediction:', baseline_predict, '\n')

        ## Baseline2
        baseline_score = scoreCal(test_input, test_target, test_output, variation=[1,1,1,1,1])
        print('baseline2 score:', baseline_score)

        ## SVM
        svm = SVM()
        # svm.fit(train_input,train_target)
        # svm.save()
        svm.load()
        test_output = svm.predict(test_input)

        ## denormalize
        test_output = codeDenormalize(test_output, code=code)
        test_data = denorm_test_input
        test_target = denorm_test_target

        ## score
        svm_score = scoreCal(test_input, test_target, test_output)
        svm_abs_score = scoreCal(test_input, test_target, test_output, count_variation=False)
        print('svm score:', svm_score)
        print('svm abs score:', svm_abs_score)       

        ## predict
        svm_predict = svm.predict([last_data[-15:]])
        svm_predict = codeDenormalize(svm_predict, code=code)    
        print('svm prediction:', svm_predict, '\n')

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
        ann_abs_score = scoreCal(test_data,test_target,test_output, count_variation=False)
        print('ann score:', ann_score)
        print('ann abs score:',ann_abs_score)
        
        ## predict
        ann_predict = ann.predict(last_data[-15:])
        ann_predict = codeDenormalize(ann_predict, code=code)        
        print('ann prediction:', ann_predict, '\n')

        ## LSTM
        future = 4
        lstm = LSTM()
        # lstm.fit(trainPath, n_epoch=70)
        # lstm.save()
        lstm.load()
        test_output = lstm.predictTestSet(testPath, future=future)

        ## denormalize
        test_output = codeDenormalize(test_output[:, (-future-1):], code=code)
        test_data = codeDenormalize(test_df.iloc[:,1:(-future-1)].values, code=code)
        test_target = codeDenormalize(test_df.iloc[:,(-future-1):].values, code=code)        
          
        ## score
        lstm_score = scoreCal(test_data,test_target,test_output)
        lstm_abs_score = scoreCal(test_data,test_target,test_output, count_variation=False)
        print('lstm score:', lstm_score)
        print('lstm abs score:', lstm_abs_score)
        
        ## predict
        lstm_predict = lstm.predict(last_data, future=4)
        lstm_predict = codeDenormalize(lstm_predict, code=code)
        lstm_predict = [lstm_predict[0][-5:]]   
        print('lstm prediction:', lstm_predict, '\n')

        # ## save results
        # result_file = open('result/result.txt', 'w')
        # result_file.write('code = '+str(code)+'\n') 
        # result_file.write('baseline1 score:'+str(baseline_score)+'\n')
        # result_file.write('baseline2 score:'+str(baseline_score)+'\n')
        # result_file.write('svm score:'+str(svm_score)+'\n')
        # result_file.write('svm abs score:'+str(svm_abs_score)+'\n')
        # result_file.write('ann score:'+str(ann_score)+'\n')
        # result_file.write('ann abs score:'+str(ann_abs_score)+'\n')
        # result_file.write('lstm score:'+str(lstm_score)+'\n')
        # result_file.write('lstm abs score:'+str(lstm_abs_score)+'\n')
        # result_file.write('baseline prediction:'+str(baseline_predict)+'\n')
        # result_file.write('svm prediction:'+str(svm_predict)+'\n')
        # result_file.write('ann prediction:'+str(ann_predict)+'\n')
        # result_file.write('lstm prediction:'+str(lstm_predict)+'\n')
