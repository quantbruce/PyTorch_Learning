############### 加载数据集 DataLoader

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath, delimiter=',', dtype=np.float32)
        data = np.array(data)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1])
        self.y_data = torch.from_numpy(data[:, [-1]]) # 这里-1要加括号，不然导出的维度是错的

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


filepath = r'D:\geek growing\pytorch\刘二大人\PyTorch深度学习实践\datasets\diabetes\diabetes.csv'
dataset = DiabetesDataset(filepath)
# print(dataset.len) # 758 行, 所以itertion = 758 / 32 = 24, 这就是后面 i = 0, 1, 2, ..., 24的原因


train_loader = DataLoader(dataset=dataset
                          , batch_size=32
                          , shuffle=True
                          # num_workers=2  # 这一行多进程的代码加了就会报错，暂未解决
                         )


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for i, (input, label) in enumerate(train_loader, 0):
        y_pred = model(input)
        loss = criterion(y_pred, label)
        print('eopch: ', epoch, 'i=', i, 'loss:', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




### MNIST 数据集

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


train_dataset = datasets.MNIST(root=''
                               , train=True
                               , transform=transforms.ToTensor()
                               , download=True
                               )
test_dataset = datasets.MNIST(root=''
                              , train=True
                              , transform=transforms.ToTensor()
                              , download=True
                              )

train_loader = DataLoader(
                        dataset=train_dataset
                        , batch_size=32
                        , shuffle=True
                        )

test_loader = DataLoader(dataset=test_dataset
                         , batch_size=32
                         , shuffle=True)

from batch_idx, (inputs, label) in enumerate(train_loader):
    ......












