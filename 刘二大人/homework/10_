###### The different configuration of this CNN
###### 自己设计不同的CNN来跑MNIST数据，对比全连接线性神经网络

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time


batch_size = 64
transform = transforms.Compose([   # transform主要是用来针对图像做一些各种原始化的处理，将像素数据[0, 255]处理成神经网络喜欢的0-1范围之间，并且服从正态分布的数据。
    transforms.ToTensor(),  # 将[0, 255]的值压缩到[0, 1]，把w*h*c --> c*w*h. 可以理解成单通道变成多通道。
    transforms.Normalize((0.1307, ), (0.3081)) # 均值0.1307, 标准差0.3081
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


class Net(torch.nn.Module):                 # Tips：在草稿纸上画下设计草图，再在这个部分分别填充即可。细节就是记住维度之间的一环扣一环，不要弄错
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(20, 30, kernel_size=3)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)  # 因为pooling函数是没有参数的，类似sigmoid，所以这里只需要定义一个pooling函数即可，不用分别定义三个
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=2, padding=1)  # 这里有参数，重新定义了一个以示区别
        self.linear1 = torch.nn.Linear(in_features=120, out_features=64)
        self.linear2 = torch.nn.Linear(in_features=64, out_features=32)
        self.fc = torch.nn.Linear(in_features=32, out_features=10)


    def forward(self, x):
        # Flatten data from (n, 1, 28, 28)
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling2(F.relu(self.conv2(x)))
        x = self.pooling(F.relu(self.conv3(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.fc(x)  # 再用全连接层做变换
        return x  # 最后一层不做激活，因为下面要喂给交叉熵损失的

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
    for epoch in range(10):
        train(epoch)  # 这样封装起来能保持代码的简洁性，不然主程序这一大堆，可读性较低。
        test()
