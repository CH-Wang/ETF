import pandas as pd
import numpy as np 
from sklearn import preprocessing

def data_rename(df):
    name_list = ['代碼', '日期', '中文簡稱', '開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)', '成交張數(張)']
    df.columns =['code','date','abbr','open','high','low','close','amount']
    return df

def data_normalizer(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    return df

def denormalize(df, norm_value):
    original_value = df['收盤價(元)'].values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm =[]
    for i in norm_value:
        i = i.reshape(-1,1)
        denorm.append(min_max_scaler.inverse_transform(i).reshape(-1))
    return denorm


def data_slicer(df):
    df = df.drop(['date','code','abbr','open','high','low'], axis=1)
    return df

def data_merger(df,new_df):
    data = []
    new_loc = 0
    for index, _ in df.iterrows():
        if index < df.index.max()-18:
            for i in range(20):
                next_row = df.iloc[index+i]
                data.append(next_row['close'])
            # data = data_normalizer(data)
            new_df.loc[new_loc] = data
            new_loc += 1
            data.clear()
    return new_df


def data_finalizer(df, train_df, test_df):
    df = df.dropna(axis=0, how='any')
    df = df.sample(frac = 1).reset_index(drop=True)
    max_index = df.index.max()
    cut_line = int(0.9*max_index)
    train_df = df.loc[df.index < cut_line]
    test_df = df.loc[df.index >= cut_line].reset_index(drop=True)
    return train_df, test_df

def score_cal(norm_input_list, norm_target_list, norm_output_list):

    df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'Big5')
    df = pd.DataFrame(df) 

    avg_score = 0
    for j, norm_input in enumerate(norm_input_list):
        norm_target = norm_target_list[j]
        norm_output = norm_output_list[j]
        denorm_input = denormalize(df, norm_input)
        denorm_target = denormalize(df, norm_target)
        denorm_output = denormalize(df, norm_output)

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
        avg_score += sum(weighted_score)

    avg_score = avg_score/ len(norm_input_list)
    return sum(avg_score)


if __name__ == '__main__':

    df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'Big5')
    df = pd.DataFrame(df)  
    df = data_rename(df)

    # find etf_0050
    df_0050 = df[df.code == 50].reset_index(drop=True)
    df_0050 = data_slicer(df_0050)
    df_0050 = data_normalizer(df_0050)
    # print(df_0050 .to_string())

    # reshape the data
    new_df_0050 = pd.DataFrame(columns=[str(i) for i in range(20)])
    new_df_0050 = data_merger(df_0050,new_df_0050)


    # split the data
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    train_df, test_df = data_finalizer(new_df_0050 , train_df, test_df)
    train_df.to_csv('../data/train.csv')
    test_df.to_csv('../data/test.csv')
    # print(train_df)
    # print(test_df)

