from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

def denormalize(df, norm_value):
    original_value = df['收盤價(元)'].values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm =[]
    for i in norm_value:
        i = i.reshape(-1,1)
        denorm.append(min_max_scaler.inverse_transform(i).reshape(-1))
    return denorm


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    train_df = pd.read_csv('../data/train.csv', sep=',',header=None)
    test_df = pd.read_csv('../data/test.csv', sep=',',header=None)
    input = torch.from_numpy(train_df.iloc[1:,1:-1].values)
    target = torch.from_numpy(train_df.iloc[1:,2:].values)
    test_input = torch.from_numpy(test_df.iloc[1:4,1:-1].values)
    test_target = torch.from_numpy(test_df.iloc[1:4,2:].values)
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(20):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            # print('predict:',pred[:, :-future])
            # print('future:', test_target)
            # print('test loss:', loss.item())
            y = pred.detach().numpy()

        df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'Big5')
        df = pd.DataFrame(df) 
        denorm_pred = denormalize(df, y)
        denorm_ytest = denormalize(df, test_target)

        # print(denorm_pred,'\n',denorm_ytest,'\n')

        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(red: pred, green: real)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # print('input size : ', input.size(),'\n')
        # print('y[:input.size(1)] : ' ,y[:input.size(1)],'\n')
        # print('y : ', y)
        # print('test_target : ', test_target)
        # test_data = test_target.numpy()

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            # plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(denorm_pred[0], 'r')
        draw(denorm_ytest[0], 'g')
        # draw(y[2], 'b')
        # plt.show()
        plt.savefig('./plot/predict%d.pdf'%i)
        plt.close()