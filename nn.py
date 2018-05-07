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
        self.fc1 = nn.Linear(15, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 10)
        self.fc5 = nn.Linear(10, 5)

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
        week = self.ETF.iloc[index, 1:16].tolist()
        next_week = self.ETF.iloc[index, 16:21].tolist()
        week = torch.Tensor(week)
        next_week = torch.Tensor(next_week)
        return week, next_week


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
    # print(inputs, labels)
    outputs = net(Variable(inputs))
    outputs = torch.round(outputs)
    print('inputs: \n',inputs,'\n\n outputs: \n',outputs,'\n\n')
    diff = torch.sum(torch.abs(outputs.data - labels))/5
    # print('post:',(outputs == labels))
    # print('post:',(outputs == labels).sum())
    # print('post:',(outputs == labels).sum().item())
    # correct += (outputs == labels).sum().item()
    
    # print (outputs.data, labels, diff)
    total += labels.size(0)
    print('size = ',labels.size(0),'\n\n total =',total, '\n\n')
    total_diff += diff 

diff_rate = total_diff/total
print(diff_rate.item())
# print('Accuracy of the network on the ',total, ' test examples: %d %%' % (
#     100 * correct / total))


print('end')

