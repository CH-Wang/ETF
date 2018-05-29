import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from data_loader import score_cal


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 100)
        self.lstm2 = nn.LSTMCell(100, 100)
        self.linear = nn.Linear(100, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 100, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 100, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 100, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 100, dtype=torch.double)

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

class ETFDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.ETF = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.ETF)

    def __getitem__(self, index):
        data = self.ETF.iloc[index, 1:-1].tolist()
        label = self.ETF.iloc[index, 2:].tolist()
        data = torch.Tensor(data).double()
        label = torch.Tensor(label).double()
        return data, label

class ETFtestset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.ETF = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.ETF)

    def __getitem__(self, index):
        data = self.ETF.iloc[index, 1:-5].tolist()
        label = self.ETF.iloc[index, -5:].tolist()
        data = torch.Tensor(data).double()
        label = torch.Tensor(label).double()
        return data, label

if __name__ == '__main__':

    trainset = ETFDataset('../data/train.csv')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)

    testset = ETFtestset('../data/test.csv')
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0)

    np.random.seed(0)
    torch.manual_seed(0)

    net = Sequence()
    net.double()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.2)

    for epoch in range(15): 

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), './model/LSTM_model')


    score = 0
    total = 0
    for data in testloader:
        inputs, labels = data
        outputs = net(Variable(inputs),future = 4)

        inputs_data = inputs.detach().numpy()
        outputs_data = outputs.detach().numpy()[:,-5:]
        labels_data = labels.detach().numpy()

        score += score_cal(inputs_data, labels_data, outputs_data)
        total +=1

    print ('avg score = ', (score/total))

    print('end')



