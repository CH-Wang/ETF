import pandas as pd
import numpy as np 

def data_normalizer(data_list):
    ## use the mean of first five days as base value
    base = 0
    for i in data_list[0:5]:
        base += i
    base = base/5
    # data_list = [round((i-base)/base, 2)  for i in data_list]
    data_list = [round((i-base)/2, 2)  for i in data_list]
    data_list[5:10] = data_list[0:5]
    return data_list 


def data_slicer(df):
    df = df.drop(['日期','代碼','中文簡稱','開盤價(元)','最高價(元)','最低價(元)','成交張數(張)'], axis=1)
    return df

def data_merger(df,new_df):
    data = []
    new_loc = 0
    base = df.iloc[0,0]
    # print(base)
    for index, row in df.iterrows():
        if index < df.index.max()-8:
            for i in range(10):
                next_row = df.iloc[index+i]
                if (next_row['收盤價(元)'] > base):
                    market = 1
                elif (next_row['收盤價(元)'] < base):
                    market = -1
                else:
                    market = 0
                base = next_row['收盤價(元)']
                data.append(market)
            # data = data_normalizer(data)
            # temp = data[0]
            # temp2 = data[5]
            # data[0:5] = [temp,temp,temp,temp,temp]
            # data[5:10] =[temp,temp,temp,temp,temp]
            new_df.loc[new_loc] = data
            new_loc += 1
            data.clear()
    return new_df


def data_finalizer(df, train_df, test_df):
    df = df.dropna(axis=0, how='any')
    df = df.sample(frac = 1).reset_index(drop=True)
    max_index = df.index.max()
    cut_line = int(0.8*max_index)
    train_df = df.loc[df.index < cut_line]
    test_df = df.loc[df.index >= cut_line].reset_index(drop=True)
    return train_df, test_df

df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'Big5')
df = pd.DataFrame(df)  


# find etf_0050
df_0050 = df[df.代碼 == 50]


df_0050 = data_slicer(df_0050)
# print(df_0050)


# print(week_df_0050)

new_df_0050 = pd.DataFrame(columns=['Mon','Tue','Wed','Thr','Fri','NMon','NTue','NWed','NThr','NFri'])
new_df_0050 = data_merger(df_0050,new_df_0050)

train_df = pd.DataFrame(columns=['Mon','Tue','Wed','Thr','Fri','NMon','NTue','NWed','NThr','NFri'])
test_df = pd.DataFrame(columns=['Mon','Tue','Wed','Thr','Fri','NMon','NTue','NWed','NThr','NFri'])

train_df, test_df = data_finalizer(new_df_0050 , train_df, test_df)
train_df.to_csv('../data/train.csv')
test_df.to_csv('../data/test.csv')
# print(train_df)
# print(test_df)
# print(two_week_df_0050)
