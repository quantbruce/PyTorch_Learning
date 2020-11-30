### part1

import numpy as np

y = np.array([1, 0, 0])
z = np.array([0.2, 0.1, -0.1])

y_pred = np.exp(z)/np.exp(z).sum()
print(y_pred)
loss = (-y * np.log(y_pred)).sum() # 实际中，由于这里概率最大label为1，其他都为0，所以往往直接提取概率最大的label出来算，其他label为0的项直接不管。
print(loss)


### part2 (part2的代码和part1是等价的，只是调用了pytorch的库)
import torch

y = torch.LongTensor([0]) # 这里要用长整型，[0]是指第0个分类，也就是对应第5行代码的[1, 0, 0]，只有第0个下标非0，其他都为0，所以直接提取出[0]来算，避免所有求和运算，更加高效
z = torch.Tensor([[0.2, 0.1, -0.1]])
criterion = torch.nn.CrossEntropyLoss() # 这一步就包含了计算exp()和loss
loss = criterion(z, y)
print(loss)


### part3
import torch

criterion = torch.nn.CrossEntropyLoss() # 要理解交叉损失的计算轨迹，本质：CrossEntropyLoss = log(softmax) + NLLLoss 
Y = torch.LongTensor([2, 0, 1]) # 原真实标签分别是 2,0,1 对应下面第一行、第二行、第三行
                #输出标签  0    1    2
Y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],  # 2的概率最大, 所以输出预测为2，真实标签为2，一致。
                        [1.1, 0.1, 0.2],  # 0的概率最大, 所以输出预测为0，真实标签为0，一致。
                        [0.2, 2.1, 0.1]]) # 1的概率最大, 所以输出预测为1，真实标签为1，一致。

Y_pred2 = torch.Tensor([[0.8, 0.2, 0.3], # 0的概率最大, 所以输出预测为0，真实标签为2，不一致。
                        [0.2, 0.3, 0.5], # 2的概率最大, 所以输出预测为2，真实标签为0，不一致。
                        [0.2, 0.2, 0.5]]) # 2的概率最大, 所以输出预测为2，真实标签为1，不一致。  # 可以推断Y_pred2的损失函数必然要大很多

l1 = criterion(Y_pred1, Y)
l2 = criterion(Y_pred2, Y)
print('Batch Loss1 = ', l1.data, '\nBatch Loss2 = ', l2.data)


### part4 MNIST手写数据集
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


batch_size = 64
transform = transforms.Compose([   # transform主要是用来针对图像做一些各种原始化的处理，将像素数据[0, 255]处理成神经网络喜欢的0-1范围之间，并且服从正态分布的数据。
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081))
])

filepath = r'D:\geek growing\pytorch\刘二大人\PyTorch深度学习实践\datasets'

train_dataset = datasets.MNIST(root=filepath
                               , train=True
                               , download=True
                               , transform=transform
                               )

train_loader = DataLoader(train_dataset
                          , shuffle=True
                          , batch_size=batch_size)

test_dataset = datasets.MNIST(root=filepath
                              , train=False
                              , download=True
                              , transform=transform)

test_loader = DataLoader(test_dataset
                         , shuffle=False
                         , batch_size=batch_size)

# print(train_dataset)
# print(test_dataset)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print('[epochs: %d, batch_idx: %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0


def test():
    correct, total = 0, 0
    with torch.no_grad(): # test_data上不需要反向求梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 返回两个，最大值和最大值的下标. 输出的predicted是个Tensor类型
           # res.extend(predicted.numpy())   可以新建个res， 用来保存预测输出的标签，但这个extend是简单拼接，其实是部队的。换做多次计算取均值的思路更合适。
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total) )


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
        
        
