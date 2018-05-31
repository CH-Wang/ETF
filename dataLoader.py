import pandas as pd
import numpy as np 
from sklearn import preprocessing

def rename(df):
#    name_list = ['代碼', '日期', '中文簡稱', '開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)', '成交張數(張)']
    df.columns =['code','date','abbr','open','high','low','close','amount']
    
    for column in df.columns[3:]:
        df[column] = df[column].apply(lambda x: float(str(x).replace(',','')))

    return df

def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    return df

def denormalize(df, norm_value):
    df = rename(df)
    original_value = df['close'].values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm =[]
    for i in norm_value:
        i = i.reshape(-1,1)
        denorm.append(min_max_scaler.inverse_transform(i).reshape(-1))
    return denorm


def dropCol(df):
    df = df.drop(['date','code','abbr','open','high','low'], axis=1)
    return df

def concat(df, ndays=20):
    newDf = pd.DataFrame(columns=[str(i) for i in range(ndays)])
    data = []
    new_loc = 0
    for index, _ in df.iterrows():
        if index < df.index.max() - ndays + 1 :
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
    df = df.sample(frac = 1).reset_index(drop=True)
    max_index = df.index.max()
    cut_line = int(0.9*max_index)
    train_df = df.loc[df.index < cut_line]
    test_df = df.loc[df.index >= cut_line].reset_index(drop=True)
    return train_df, test_df

def shift(df):
    df.close = df.close - df.close.shift()
    return df


def scoreCal(norm_input_list, norm_target_list, norm_output_list):

    df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'cp950')
    df = pd.DataFrame(df) 

    avg_score = 0
    for j, norm_input in enumerate(norm_input_list):
        norm_target = norm_target_list[j]
        norm_output = norm_output_list[j]
        denorm_input = denormalize(df, norm_input)
        denorm_target = denormalize(df, norm_target)
        denorm_output = denormalize(df, norm_output)

        score_list = []
        last_target_price = denorm_input[-1].round(2)
        last_output_price = denorm_input[-1].round(2)
        for i in range(5):
            score = 0
            target_price = denorm_target[i].round(2)
            output_price = denorm_output[i].round(2)
            diff_target = target_price - last_target_price
            diff_ouput = output_price - last_output_price           
            last_target_price = target_price
            last_output_price = output_price

            if (diff_target*diff_ouput > 0 or diff_target == diff_ouput):
                score += 0.5
            score += (target_price - abs(target_price - output_price))/target_price*0.5
            score_list.append(score)
        weighted_score = map(lambda x,y:x*y,score_list,[0.1,0.15,0.2,0.25,0.3])
        avg_score += sum(weighted_score)

    avg_score = avg_score/ len(norm_input_list)
    return sum(avg_score)


if __name__ == '__main__':

    df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'cp950')
    df = pd.DataFrame(df)  
    df = rename(df)

    ETFcode = [50,51,52,53,54,55,56,57,58,59,6201,6203,6204,6208,690,692,701,713]
    
    for code in [50]:

        codeDf = df[df.code == code].reset_index(drop=True)

        # normalize and shift 
        # codeDf = shift(codeDf)
        codeDf = codeDf.dropna(axis=0, how='any')
        codeDf = normalize(codeDf)
        
        # concatenate data for 20 days
        codeDf = concat(codeDf)
        # print(codeDf.to_string())

        # split the data
        train_df, test_df = split(codeDf)
        train_df.to_csv('../data/train.csv')
        test_df.to_csv('../data/test.csv')
        # print(train_df)
        # print(test_df)

