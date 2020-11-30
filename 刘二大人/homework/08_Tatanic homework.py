##### Tatanic homework

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
pd.set_option('display.max_rows', 1000)

filepath = 'D:/geek growing/pytorch/刘二大人/PyTorch深度学习实践/homework/Lecture_08_Dataset_and_Dataloader/'


class TatanicDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath, delimiter=',')  # , dtype=np.float32
        data = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
        data = data.fillna(0).astype('float32')
        data = np.array(data)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1])
        self.y_data = torch.from_numpy(data[:, [-1]])

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len



train_dataset = TatanicDataset(filepath + 'train.csv')


train_loader = DataLoader(
                        dataset=train_dataset
                        , batch_size=32
                        , shuffle=True
                        )

# test_loader = DataLoader(
#                         dataset=
#
#                         )


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Liner1 = torch.nn.Linear(5, 3)
        self.Liner2 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.Liner1(x))
        x = self.sigmoid(self.Liner2(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(100):
    for batch_idx, (input, label) in enumerate(train_loader, 0):
        y_pred = model(input)  # model 已经训练出来了
        # train_pred += (y_pred.detach().numpy().ravel())/100  # 学到一点，如果原来tensor有包含梯度，则要先.detach().numpy()才可行，不然会报错。
        loss = criterion(y_pred, label)
        print('epch: ', epoch, 'batch_idx: ', batch_idx, 'loss: ', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print(train_pred)
    # exit()
