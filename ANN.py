import torch
import torchvision
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from dataLoader import scoreCal
from loss import ETFLoss



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(15, 50, bias=True)
        self.fc2 = nn.Linear(50, 50, bias=True)
        self.fc3 = nn.Linear(50, 50, bias=True)
        self.fc4 = nn.Linear(50, 10, bias=True)
        self.fc5 = nn.Linear(10, 5, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class ETFDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.ETF = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.ETF)

    def __getitem__(self, index):
        data = self.ETF.iloc[index, 1:16].tolist()
        label = self.ETF.iloc[index, 16:].tolist()
        data = torch.Tensor(data)
        label = torch.Tensor(label)
        return data, label


class ANN():
    def __init__(self):
        self.model = Net()

    def fit(self, trainpath, n_epoch = 15, lr = 0.001, batch_size=4):
        
        trainset = ETFDataset(trainpath)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size,shuffle=True, num_workers=0)
        

        criterion = ETFLoss()
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

    def predict(self, test_input):
        test_input= torch.Tensor(np.array([test_input]))
        return self.model(Variable(test_input)).detach().numpy()

    def save(self, code = 50):
        filename = 'ANN'+ str(code)
        filepath = './model/' + filename
        torch.save(self.model.state_dict(), filepath)

    def load(self, code = 50):
        filename = 'ANN'+ str(code)
        filepath = './model/' + filename
        self.model.load_state_dict(torch.load(filepath))

    def predictTestSet(self, testpath):
        testset = ETFDataset(testpath)
        testloader = torch.utils.data.DataLoader(testset,shuffle=False)
        output_list = []
        for data in testloader:
            inputs, labels = data
            outputs = self.model(Variable(inputs))
            output_list.append(outputs.detach().numpy())
        return output_list

