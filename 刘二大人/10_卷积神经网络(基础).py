#############CNN 基础

### part1 观察下维度
import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size
                    , in_channels
                    , width
                    , height
                    )

conv_layer = torch.nn.Conv2d(in_channels
                             , out_channels
                             , kernel_size=kernel_size
                             )

output = conv_layer(input)


print('input.shape:', input.shape)
print('output.shape:', output.shape)
print('conv_layer.weight.shape:', conv_layer.weight.shape) # 这个结果(输出通道，输入通道，卷积核W, 卷积核H)



#### part2

import torch

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]

input = torch.Tensor(input).view(1, 1, 5, 5)

con_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
print(con_layer.weight.data)
con_layer.weight.data = kernel.data
print(con_layer.weight.data)

ouput = con_layer(input)
print(ouput)


##### part3

import torch

input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6]

input = torch.Tensor(input).view(1, 1, 4, 4)
max_pooling_layer = torch.nn.MaxPool2d(kernel_size=2) # 默认stride = 2

output = max_pooling_layer(input)
print(output)


#### part4
import torch
from torch.nn import functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(in_features=320, out_features=10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28)
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)  # 再用全连接层做变换
        return x  # 最后一层不做激活，因为下面要喂给交叉熵损失的

model = Net()


##### part5(上接part4) 模型迁移到GPU

# send model to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
print(model)

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
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0
            
            
#### part6 (part5的完整代码实现)

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(in_features=320, out_features=10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28)
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)  # 再用全连接层做变换
        return x  # 最后一层不做激活，因为下面要喂给交叉熵损失的

model = Net()


# send model to GPU
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
    print('Accuary on test set: [%d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)  # 这样封装起来能保持代码的简洁性，不然主程序这一大堆，可读性较低。
        test()




