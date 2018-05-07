import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

import pandas as pd

import numpy as np



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 20, bias=True)
        self.fc2 = nn.Linear(20, 20, bias=True)
        self.fc3 = nn.Linear(20, 20, bias=True)
        self.fc4 = nn.Linear(20, 10, bias=True)
        self.fc5 = nn.Linear(10, 1, bias=True)

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
        data = self.ETF.iloc[index, 1:20].tolist()
        label = self.ETF.iloc[index, 20:21].tolist()
        data = torch.Tensor(data)
        label = torch.Tensor(label)
        return data, label


transformations = transforms.Compose([transforms.ToTensor()])

trainset = ETFDataset('../data/train.csv')


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)

testset = ETFDataset('../data/test.csv')


testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0)


net = Net()


criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.2)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # print(outputs, labels, loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), './model/model')


diff_rate = 0
total_diff = 0
total = 0
correct = 0
for data in testloader:
    inputs, labels = data
    outputs = net(Variable(inputs))
    # outputs = torch.round(outputs)

    print('inputs: \n',inputs,'\n\n outputs: \n',outputs,'\n\n')
    correct += (outputs == labels).sum().item()

    total += labels.size(0)
    print('size = ',labels.size(0),'\n\n total =',total, '\n\n')

print('Accuracy of the network on the ',total, ' test examples: %d %%' % (
    100 * correct / total))


print('end')

