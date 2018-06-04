import pandas as pd
import numpy as np 
from sklearn import preprocessing

def rename(df):
##    name_list = ['代碼', '日期', '中文簡稱', '開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)', '成交張數(張)']
    df.columns=['code','date','abbr','open','high','low','close','amount']
    
    for column in df.columns[3:]:
        df[column] = df[column].apply(lambda x: float(str(x).replace(',','')))

    return df

def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    return df

def denormalize(df, norm_value):
    """
        denormalize the norm_value using the "close" price of the dataframe 
        norm_value type: list with dim(n,n), [[float,float,float,..], [float,float,float,..],...]
    """
    original_value = df['close'].values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm =[]
    for i in norm_value:
        i = i.reshape(-1,1)
        denorm.append(min_max_scaler.inverse_transform(i).reshape(-1))
    return denorm

def codeDenormalize(data_list, code = 50, filepath = '../data/TBrain_Round2_DataSet_20180601/tetfp.csv', encoding = 'cp950'):
    """
        deciding the dataframe for denormalize,
    """   
    df = pd.read_csv(filepath, encoding=encoding)
    df = pd.DataFrame(df)
    df = rename(df)
    df = df[df.code == code].reset_index(drop=True) 
    return  denormalize(df, data_list)

def dropCol(df):
    df = df.drop(['date','code','abbr','open','high','low'], axis=1)
    return df

def concat(df, ndays=20):
    """
        concatenate the "close" price for days = ndays
    """
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
    """
        Calculate ETF score using the "denormalized" data
        data_list records the input data for prediction,
        target_list records the label for input,
        output_list records the prediction
        data_list type : list with dim(n,n), [[float,float,float,...,], [float,float,float,...,],...]
        target_list type: list with dim(n,5), [[float,float,float,float,float], [float,float,float,float,float],...]
        output_list type: list with dim(n,5), [[float,float,float,float,float], [float,float,float,float,float],...]
        variaion is an optional parameter, by default the vairation score is calculate
        by the predicted price, use this parameter to calculate varation score by customize
        predictions.
        variation type: int list with dim(5),[0,1,0,-1,0]
        count_variation decideds whether to count the variation score
    """

    avg_score = 0
    for j, data in enumerate(data_list):
        target = target_list[j]
        output = output_list[j]
        score_list = []
        last_target_price = data[-1]
        last_output_price = data[-1]
        
        for i in range(5):
            score = 0
            target_price = target[i]
            output_price = output[i]
            
            ## calculate variation score
            if count_variation:
                var_target = np.sign(target_price - last_target_price)
                var_ouput = np.sign(output_price - last_output_price)           
                last_target_price = target_price
                last_output_price = output_price
                if variation:
                    if (var_target == variation[i]):
                        score += 0.5                   
                elif (var_target == var_ouput):
                    score += 0.5
            
            ## calculate absolute price score
            score += (target_price - abs(target_price - output_price))/target_price*0.5
            score_list.append(score)
        weighted_score = map(lambda x,y:x*y,score_list,[0.1,0.15,0.2,0.25,0.3])
        avg_score += sum(weighted_score)

    avg_score = avg_score/ len(data_list)
    return avg_score


if __name__ == '__main__':

    filepath = '../data/TBrain_Round2_DataSet_20180601/tetfp.csv'
    df = pd.read_csv(filepath,encoding = 'cp950')
    df = pd.DataFrame(df)  
    df = rename(df)

    ETFcode = [50,51,52,53,54,55,56,57,58,59,6201,6203,6204,6208,690,692,701,713]
    
    for code in ETFcode:

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

        trainDf.to_csv(trainPath, index=False)
        testDf.to_csv(testPath, index=False)


