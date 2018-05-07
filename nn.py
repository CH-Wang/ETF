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
from data_loader import denormalize



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 50, bias=True)
        self.fc2 = nn.Linear(50, 50, bias=True)
        self.fc3 = nn.Linear(50, 50, bias=True)
        self.fc4 = nn.Linear(50, 10, bias=True)
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

criterion = nn.MSELoss()
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
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), './model/model')


diff_rate = 0
total_diff = 0
total = 0
correct = 0
df = pd.read_csv('../data/TBrain_Round2_DataSet_20180331/tetfp.csv',encoding = 'Big5')
df = pd.DataFrame(df)
for data in testloader:
    inputs, labels = data
    outputs = net(Variable(inputs))

    # correct += (outputs == labels).sum().item()

    inputs_data = inputs.detach().numpy()
    outputs_data = outputs.detach().numpy()
    labels_data = labels.detach().numpy()

    
    for i in range(len(outputs_data)):
        base = inputs_data[i][-1]
        outputs_val = outputs_data[i][0]
        labels_val = labels_data[i][0]

        if (outputs_val > base) : pre_state = 1
        elif (round(outputs_val,2) == round(base,2)) : pre_state = 0
        else : pre_state = -1

        if (labels_val > base) : state = 1
        elif (round(labels_val,4) == round(base,4)) : state = 0
        else : state = -1
        if (state == pre_state): correct += 1

    denorm_pred = denormalize(df, outputs_data)
    denorm_test = denormalize(df, labels_data)
    
    err = 0
    for i in range(len(denorm_pred)):
        err += denorm_pred[i] - denorm_test[i]
    total_diff += err

    total += labels.size(0)

print ('error rate = ', (total_diff/total)[0])
print('correct change rate on the ',total, ' test examples: %.2f' % (correct / total))


print('end')

