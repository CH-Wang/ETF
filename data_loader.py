import pandas as pd
import numpy as np 

def datetime_conversion(df):
    df['日期']=pd.to_datetime(df.iloc[:,1], format='%Y%m%d')
    return df

## index: monday ==0, friday == 4
def weekday_conversion(df):
    df['星期'] = df['日期'].dt.weekday
    return df

## add a new column '週數'
def week_num_conversion(df):
    week_count_list = []
    next_week_list = []
    base_time = pd.to_datetime(df.iloc[0,1], format='%Y%m%d')
    base_time_weekday = base_time.weekday()
    if base_time_weekday != 0:
        base_time = base_time - pd.to_timedelta(base_time_weekday, unit='D')
    for date in df['日期']:
        date_diff = date - base_time
        week_num = int(date_diff/np.timedelta64(7, 'D'))
        week_count_list.append(week_num)
        next_week_list.append(week_num+1)
    df['週數'] = week_count_list
    df['下週'] = next_week_list
    return df
     
def data_slicer(df):
    df = df.drop(['日期','代碼','中文簡稱','開盤價(元)','最高價(元)','最低價(元)','成交張數(張)'], axis=1)
    return df

def week_data_merger(df,week_df):
    week = 0
    week_data = [float('nan'), float('nan'),float('nan'),float('nan'),float('nan'),week]
    new_loc = 0
    for index, row in df.iterrows():
        if int(row['週數']) == week:
            ## Ignore Sat & Sun's stock market
            if(int(row['星期'])<5):
                week_data[int(row['星期'])] = row['收盤價(元)']
        else:
            week_df.loc[new_loc] = week_data
            new_loc += 1
            week = int(row['週數'])
            week_data = [float('nan'), float('nan'),float('nan'),float('nan'),float('nan'),week]
            if(int(row['星期'])<5):
                week_data[int(row['星期'])] = row['收盤價(元)']
    return week_df

def two_week_data_merger(week_df,two_week_df):
    week = week_df.iloc[0,5]
    two_week_data = []
    new_loc = 0
    for index, row in week_df.iterrows():
        if index < week_df.index.max():
            next_row = week_df.iloc[index+1]
            if next_row['week'] == row['week']+1:
                two_week_data = row.tolist()
                two_week_data += next_row.tolist()
                two_week_df.loc[new_loc] = two_week_data 
                new_loc += 1
        two_week_data.clear()
    return two_week_df

def data_finalizer(df, train_df, test_df):
    df = df.dropna(axis=0, how='any')
    df = df.sample(frac = 1).reset_index(drop=True)
    df = df.drop(['week','Nweek'],axis=1)
    max_index = df.index.max()
    cut_line = int(0.8*max_index)
    train_df = df.loc[df.index < cut_line]
    test_df = df.loc[df.index >= cut_line].reset_index(drop=True)
    return train_df, test_df

df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'Big5')
df = pd.DataFrame(df)  

## modify data's date and weekdate
df = datetime_conversion(df)
## index: monday ==0, friday == 4
df = weekday_conversion(df)

## find etf_0050
df_0050 = df[df.代碼 == 50]

df_0050 = week_num_conversion(df_0050)


df_0050 = data_slicer(df_0050)
# print(df_0050)

df = df_0050 
week_df_0050 = pd.DataFrame(columns=['Mon','Tue','Wed','Thr','Fri','week'])
week_df_0050 = week_data_merger(df_0050,week_df_0050)

# print(week_df_0050)

two_week_df_0050 = pd.DataFrame(columns=['Mon','Tue','Wed','Thr','Fri','week','NMon','NTue','NWed','NThr','NFri','Nweek'])
two_week_df_0050 =two_week_data_merger(week_df_0050,two_week_df_0050)

train_df = pd.DataFrame(columns=['Mon','Tue','Wed','Thr','Fri','NMon','NTue','NWed','NThr','NFri'])
test_df = pd.DataFrame(columns=['Mon','Tue','Wed','Thr','Fri','NMon','NTue','NWed','NThr','NFri'])

train_df, test_df = data_finalizer(two_week_df_0050 , train_df, test_df)
train_df.to_csv('../data/train.csv')
test_df.to_csv('../data/test.csv')
# print(train_df)
# print(test_df)
# print(two_week_df_0050)




