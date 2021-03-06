import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from dataLoader import scoreCal
from dataLoader import codeDenormalize
from loss import ETFLSTMLoss

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 60)
        self.lstm2 = nn.LSTMCell(60, 60)
        self.linear = nn.Linear(60, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 60, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 60, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 60, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 60, dtype=torch.double)

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
        data = self.ETF.iloc[index, :-1].tolist()
        label = self.ETF.iloc[index, 1:].tolist()
        data = torch.Tensor(data).double()
        label = torch.Tensor(label).double()
        return data, label

class LSTM():
    def __init__(self):
        self.model = Sequence().double()

    def fit(self, trainpath, n_epoch=15, lr=0.001, batch_size=4):
        
        trainset = ETFDataset(trainpath)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size,shuffle=True, num_workers=0)
        
        np.random.seed(0)
        torch.manual_seed(0)
        criterion = ETFLSTMLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(n_epoch): 
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels, inputs)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data
                if i % 10 == 9:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                        (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished Training')

    def predict(self, test_input, future = 0):
        test_input= torch.from_numpy(np.array([test_input]))
        return self.model(test_input, future = future).detach().numpy()

    def save(self, code = 50):
        filename = 'LSTM'+ str(code)
        filepath = './model/' + filename
        torch.save(self.model.state_dict(), filepath)

    def load(self, code = 50):
        filename = 'LSTM'+ str(code)
        filepath = './model/' + filename
        self.model.load_state_dict(torch.load(filepath))

    def predictTestSet(self, testpath, future=4):
        df = pd.read_csv(testpath)
        if future <= 0:
            record_list = torch.from_numpy(df.iloc[:,:].values)       
        else:
            record_list = torch.from_numpy(df.iloc[:,:(-future-1)].values)
        return self.model(record_list, future = future).detach().numpy()    
    


