import pandas as pd
import numpy as np 
from sklearn import preprocessing

def rename(df):
##    name_list = ['代碼', '日期', '中文簡稱', '開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)', '成交張數(張)']
    df.columns =['code','date','abbr','open','high','low','close','amount']
    
    for column in df.columns[3:]:
        df[column] = df[column].apply(lambda x: float(str(x).replace(',','')))

    return df

def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    return df

def denormalize(df, norm_value):
## norm_value : [[1,0,1,..], [0,1,0,..],...]
    original_value = df['close'].values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm =[]
    for i in norm_value:
        i = i.reshape(-1,1)
        denorm.append(min_max_scaler.inverse_transform(i).reshape(-1))
    return denorm

def codeDenormalize(data_list, code = 50, filepath = '../data/TBrain_Round2_DataSet_20180331/tetfp.csv', encoding = 'cp950'):
    df = pd.read_csv(filepath, encoding=encoding)
    df = pd.DataFrame(df)
    df = rename(df)
    df = df[df.code == code].reset_index(drop=True) 
    return  denormalize(df, data_list)

def dropCol(df):
    df = df.drop(['date','code','abbr','open','high','low'], axis=1)
    return df

def concat(df, ndays=20):
    newDf = pd.DataFrame(columns=[str(i) for i in range(ndays)])
    data = []
    new_loc = 0
    for index, _ in df.iterrows():
        if index < df.index.max() - ndays + 2 :
            for i in range(ndays):
                next_row = df.iloc[index+i]
                data.append(next_row['close'])
            newDf.loc[new_loc] = data
            new_loc += 1
            data.clear()
    df = newDf
    return df


def split(df):
    df = df.dropna(axis=0, how='any')
    last_row = df.iloc[-1]
    df = df[:-1]
    df = df.sample(frac = 1).reset_index(drop=True)
    max_index = df.index.max()
    cut_line = int(0.9*max_index)
    trainDf = df.loc[df.index < cut_line]
    testDf = df.loc[df.index >= cut_line].reset_index(drop=True)
    testDf = testDf.append(last_row, ignore_index=True)
    return trainDf, testDf

def shift(df):
    df.close = df.close - df.close.shift()
    return df

def scoreCal(data_list, target_list, output_list, variation = [], count_variation = True):
## data_list : [[1,2,3,4,5], [1,2,3,4,5],...]
    avg_score = 0
    for j, data in enumerate(data_list):
        target = target_list[j]
        output = output_list[j]
        score_list = []
        last_target_price = data[-1].round(2)
        last_output_price = data[-1].round(2)
        
        for i in range(5):
            score = 0
            target_price = target[i].round(2)
            output_price = output[i].round(2)
            
            ## calculate variation score
            if count_variation:
                diff_target = target_price - last_target_price
                diff_ouput = output_price - last_output_price           
                last_target_price = target_price
                last_output_price = output_price
                if variation:
                    if (diff_target*variation[i] > 0 or diff_target == diff_ouput):
                        score += 0.5                   
                elif (diff_target*diff_ouput > 0 or diff_target == diff_ouput):
                    score += 0.5
            
            ## calculate absolute price score
            score += (target_price - abs(target_price - output_price))/target_price*0.5
            score_list.append(score)
        weighted_score = map(lambda x,y:x*y,score_list,[0.1,0.15,0.2,0.25,0.3])
        avg_score += sum(weighted_score)

    avg_score = avg_score/ len(data_list)
    return avg_score


if __name__ == '__main__':

    df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'cp950')
    df = pd.DataFrame(df)  
    df = rename(df)

    ETFcode = [50,51,52,53,54,55,56,57,58,59,6201,6203,6204,6208,690,692,701,713]
    
    for code in [50]:

        codeDf = df[df.code == code].reset_index(drop=True)

        ## normalize and shift 
        # codeDf = shift(codeDf)
        codeDf = codeDf.dropna(axis=0, how='any')
        codeDf = normalize(codeDf)

        ## concatenate data for 20 days
        codeDf = concat(codeDf)

        ## split the data
        trainDf, testDf = split(codeDf)

        trainPath = '../data/train'+str(code)+'.csv'
        testPath = '../data/test'+str(code)+'.csv'

        trainDf.to_csv(trainPath)
        testDf.to_csv(testPath)


