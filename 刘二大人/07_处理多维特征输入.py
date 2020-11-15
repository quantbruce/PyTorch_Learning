import pandas as pd
import numpy as np
import torch

# 1.构建数据集

data = pd.read_csv(r'D:\geek growing\pytorch\刘二大人\PyTorch深度学习实践\diabetes.csv\diabetes.csv', delimiter=',', dtype=np.float32)
# print(data)
# xy = np.loadtxt(r'E:\Anaconda\Anaconda\Lib\site-packages\sklearn\datasets\data\diabetes_data.csv.gz', delimiter=',', dtype=np.float32)
data = np.array(data)
# print(data)
x_data = torch.from_numpy(data[:, :-1])
y_data = torch.from_numpy(data[:, [-1]])
# print(x_data)
# print(y_data)
# print(x_data.shape, y_data.shape)


# 2.建立类和模型

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

# 3. 构建损失函数和优化器

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# 4. 训练模型

for epoch in range(100):
    # forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('epoch: ', epoch, 'loss: ', loss.item())
    # backward
    optimizer.zero_grad()
    loss.backward()
    # updata
    optimizer.step()
    
