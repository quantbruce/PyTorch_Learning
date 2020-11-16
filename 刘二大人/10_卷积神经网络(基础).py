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
