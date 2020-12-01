#########################################otto-group-product-classification-challenge（Pytorch处理多分类问题）#######################################################################################
URL: 
https://www.kaggle.com/c/otto-group-product-classification-challenge/notebooks
reference: 
https://github.com/conradwhiteley/Otto-Neural-Net/blob/master/Neural_Network.ipynb  # 可以学习如何利用谷歌colab远端服务器，GPU加速大大计算速度
https://www.cnblogs.com/heyour/p/13466077.html


###### Import

import numpy as np
import pandas as pd
import os
import time
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import  Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim


###### Prepare

# 处理数据
filepath = 'D:/geek growing/pytorch/刘二大人/PyTorch深度学习实践/homework/Lecture_09_Softmax_Classifier/'
csv_path = os.path.join(filepath + 'train.csv')
datas = pd.read_csv(csv_path)
datas = datas.copy()
datas = datas.drop(columns='id')
datas = datas.sample(frac=1)  # 就是打乱顺序而已, 相当于shuffle=True

datas.target = datas.target.astype('category').cat.codes # .astype('category').cat.codes


### 划分数据集
rows, _ = datas.shape
train_rows = int(rows*0.7)

train_datas = datas.iloc[:train_rows, :]
val_datas = datas.iloc[train_rows:, :]

#### 得到features和targets
train_features =  train_datas.iloc[:, train_datas.columns != 'target'].values
train_targets = train_datas.target.values

val_features = val_datas.iloc[:, val_datas.columns != 'target'].values
val_targets = val_datas.target.values


###### 分装dataset

class CustomDataset(Dataset):
    def __init__(self, features, targets):
        super(CustomDataset, self).__init__()
        self.features = features
        self.target = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item], self.target[item]


### 得到Dataset类
train_ds = CustomDataset(train_features, train_targets)
val_ds = CustomDataset(val_features, val_targets)


### 得到DataLoader
batch_size = 64
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size)
# # print(len(train_loader)) # 该长度等于 len(train_data)/batch_size = 43314 / 677
# # print(len(train_loader)) # 677
# # print(len(train_ds)) # 43314


### Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##### 设计神经网络模型

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(93, 64)
        self.l2 = torch.nn.Linear(64, 48) # 模型可以更加复杂些, 加批量正则化，dropout等处理
        self.l3 = torch.nn.Linear(48, 24)
        self.l4 = torch.nn.Linear(24, 16)
        self.l5 = torch.nn.Linear(16, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


### 实例化模型
model = Net().to(device)


### 构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

## 封装训练函数和测试函数
def train():
    running_loss = 0.0
    accs = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        accs += (predicted == targets).sum().item()
    running_loss /= len(train_loader)
    accs = accs / len(train_ds) * 100
    print('Training: loss: %.2f accuary: %.2f%%' % (running_loss, accs))
    return running_loss, accs


def val():
    loss = 0.0
    accs = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs.float())
            loss += criterion(outputs, targets.long()).item()
            _, predicted = torch.max(outputs.data, dim=1)
            accs += (predicted == targets).sum().item()
    loss /= len(val_loader)
    accs = accs / len(val_ds) * 100
    print('Validating: loss: %.2f  accuary: %.2f%%' % (loss, accs))
    return loss, accs


##### 开始运算
train_losses = []
val_losses = []
train_accs = []
val_accs = []
maxAcc = 0


for epoch in range(100):
    print('epoch %d: ' % (epoch+1))
    losses, accs = train()
    train_losses.append(losses)
    train_accs.append(accs)

    losses, accs = val()
    val_losses.append(losses)
    val_accs.append(accs)

    if maxAcc < accs:
       maxAcc = accs

    check = np.greater(maxAcc, val_accs[-10:]) # 后面连续10个单调递减，基本可以断定已经达到最佳训练epochs了, 此时可停止训练
    if (check.all() == True) and (epoch > 25):
        print('Convergence meet')
        break

print('Maximum validation accuary %.2f%%' % maxAcc)


### 绘图 (采用了指数平滑)

# print(train_losses)
# print(val_losses)
tmp_train_losses = [math.exp(i) for i in train_losses]
tmp_val_losses = [math.exp(i) for i in val_losses]

plt.figure(figsize=[10, 5])
plt.plot(tmp_train_losses, 'r', label='Training')
plt.plot(tmp_val_losses, 'b', label='Validating')
plt.title('Network Model Work Performance on TrainSet and ValidSet')
plt.xlabel('Epoch')
plt.ylabel('Loss per epoch')
plt.legend()
plt.show()
