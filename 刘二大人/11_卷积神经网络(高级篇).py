########## class11 实现GoogleNet
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

####################################################################################################### 代码不同之处在以下框起来部分, 好好琢磨清楚

class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1) # branch_pool只是代表分支的命名而已，根据那个InceptionA结构图，实际是先进行卷积运算

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)   # B,C,W,H dim=1就是按照通道C的方向去拼接


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5) # InceptionA中各个分支输出通道的和 24 + 16 + 24 + 24 = 88

        self.incep1 = InceptionA(in_channels=10) # 不管InceptionA的输入通道是多少，输出通道都是 88 维
        self.incep2 = InceptionA(in_channels=20) # InceptionA的作用也仅仅只能改变通道C的维度，其他几个维度保持不变

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)   # (1)     # 这个1408是根据MNIST数据集维度计算出来的

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1) # (2)      # 这个算法pycharm本地上跑可以跑出准确率上99%, 比之前98%，97%有较大提升。推测影响该模型性能的主要原因在于，下一行全连接层数目太少，降维降的太猛烈。
        x = self.fc(x)          # (3)      # 1408是MNIST 28*28的数据一路运算过来 88*4*4 = 1408
        return x                           # 将(1)(2)(3)(4)(5)几个部分，分别注释和恢复后，就可以实现让计算机自动推导计算出维度(64, 88, 4, 4) 其实后三项之积就等于1408

###########################################################################################################

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

        
        
