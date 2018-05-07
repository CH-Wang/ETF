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

df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'Big5')
df = pd.DataFrame(df)  
df = data_rename(df)

# find etf_0050
df_0050 = df[df.code == 50]
df_0050 = data_slicer(df_0050)
df_0050 = data_normalizer(df_0050)

# reshape the data
new_df_0050 = pd.DataFrame(columns=[str(i) for i in range(20)])
new_df_0050 = data_merger(df_0050,new_df_0050)


# split the data
train_df = pd.DataFrame(columns=[str(i) for i in range(20)])
test_df = pd.DataFrame(columns=[str(i) for i in range(20)])

train_df, test_df = data_finalizer(new_df_0050 , train_df, test_df)
train_df.to_csv('../data/train.csv')
test_df.to_csv('../data/test.csv')
# print(train_df)
# print(test_df)

