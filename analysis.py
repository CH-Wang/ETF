import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    ## load data
    filepath = './result/result.csv'
    submissionpath = './result/submission.csv'
    ETFcode = [50,51,52,53,54,55,56,57,58,59,6201,6203,6204,6208,690,692,701,713]
    submissionColumns = ['ETFid','Mon_ud','Mon_cprice','Tue_ud','Tue_cprice','Wed_ud','Wed_cprice','Thu_ud','Thu_cprice','Fri_ud','Fri_cprice']
    
    df = pd.DataFrame()
    df = pd.read_csv(filepath, index_col=0)
    df = pd.DataFrame(df)
    
    baselineDf = df.loc[0]
    svmDf = df.loc[1]
    annDf = df.loc[2]
    lstmDf = df.loc[3]
    submissionDf = pd.DataFrame(columns = submissionColumns )

    ## calculate score for each model
    baselineScore = baselineDf['score'].sum()
    svmScore = svmDf['score'].sum()    # print('baselineScore:', baselineScore, '\n') 
    # print('svmScore:',svmScore, '\n') 
    # print('annScore:',annScore, '\n') 
    # print('lstmScore:',lstmScore, '\n') 

    annScore = annDf['score'].sum()
    lstmScore = lstmDf['score'].sum()


    ## calculate variation
    Mon_ud = pd.DataFrame(np.sign(df['1'] - df['0']) , columns=['Mon_ud'] ).astype('int64')
    Tue_ud = pd.DataFrame(np.sign(df['2'] - df['1']) , columns=['Tue_ud'] ).astype('int64')
    Wed_ud = pd.DataFrame(np.sign(df['3'] - df['2']) , columns=['Wed_ud'] ).astype('int64')
    Thu_ud = pd.DataFrame(np.sign(df['4'] - df['3']) , columns=['Thu_ud'] ).astype('int64')
    Fri_ud = pd.DataFrame(np.sign(df['5'] - df['4']) , columns=['Fri_ud'] ).astype('int64')        

    df = pd.concat([df, Mon_ud], axis=1, sort=False)
    df = pd.concat([df, Tue_ud], axis=1, sort=False)
    df = pd.concat([df, Wed_ud], axis=1, sort=False)
    df = pd.concat([df, Thu_ud], axis=1, sort=False)
    df = pd.concat([df, Fri_ud], axis=1, sort=False)

    ## submission
    decision = 1
    dfColumns = ['code', 'Mon_ud', '1', 'Tue_ud', '2', 'Wed_ud', '3', 'Thu_ud', '4', 'Fri_ud', '5']
    submissionDf[submissionColumns] = df.loc[decision][dfColumns]
    submissionDf = submissionDf.reset_index(drop=True)
    submissionDf.to_csv(submissionpath, index=False)
