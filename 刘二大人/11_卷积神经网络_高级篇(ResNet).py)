#######################ResetNet搭建
##### 准确率目前是最高的，可以较稳定达到99%


import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time

batch_size = 64
transform = transforms.Compose([  # transform主要是用来针对图像做一些各种原始化的处理，将像素数据[0, 255]处理成神经网络喜欢的0-1范围之间，并且服从正态分布的数据。
    transforms.ToTensor(),  # 将[0, 255]的值压缩到[0, 1]，把w*h*c --> c*w*h. 可以理解成单通道变成多通道。
    transforms.Normalize((0.1307,), (0.3081))  # 均值0.1307, 标准差0.3081
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


############################################################################################################ 残差网络的关键模块

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

############################################################################################################

model = Net()


#### send model to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# send the inputs and targets at every step to the GPU
# 迁移的model和input/output要保持在同一块显卡上

def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # 加了这句，就够了
        optimizer.zero_grad()
        # forward + backward + update
        output = model(inputs)
        # print(output.size())      #  (4)
        # exit()                    #  (5)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print('[epoch: %d, batch_idx: %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0

def test():
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuary on test set: [%d %% [correct/total: %d/ %d]]' % (100 * correct / total, correct, total))



if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)  # 这样封装起来能保持代码的简洁性，不然主程序这一大堆，可读性较低。
        test()
        
