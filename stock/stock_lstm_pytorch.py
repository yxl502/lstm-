import os
from builtins import type


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame
from pandas import concat
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class MyDataset(Dataset):
    def __init__(self, data_path='./shanghai_index_1990_12_19_to_2020_03_12.csv', timesteps_in=3, timesteps_out=3):
        data = pd.read_csv(data_path)
        data_set = data[['Price']].values.astype('float64')
        self.get_train_set(data_set, timesteps_in=timesteps_in, timesteps_out=timesteps_out)

    def __getitem__(self, item):
        return self.train_x[item], self.train_y[item]

    def __len__(self):
        return len(self.train_x)  # 给item一个范围

    def get_train_set(self, data_set, timesteps_in, timesteps_out=1):
        train_data_set = np.array(data_set)
        reframed_train_data_set = np.array(
            self.series_to_supervised(train_data_set, timesteps_in, timesteps_out).values)
        train_x, train_y = reframed_train_data_set[:, :-timesteps_out], reframed_train_data_set[:, -timesteps_out:]
        # 将数据集重构为符合LSTM要求的数据格式,即 [样本数，时间步，特征]
        train_x = train_x.reshape((train_x.shape[0], timesteps_in, 1))
        self.train_x = torch.from_numpy(train_x).float()
        self.train_y = torch.from_numpy(train_y).float()

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # 预测序列 (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        # 去掉NaN行
        if dropnan:
            agg.dropna(inplace=True)
        return agg


class Lstm_Model(nn.Module):
    def __init__(self):
        super(Lstm_Model, self).__init__()
        self.lstm_layer1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.lstm_layer2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.linear_layer = nn.Linear(in_features=128, out_features=3, bias=True)

    def forward(self, x):
        out1, (h_n1, h_c1) = self.lstm_layer1(x)
        out1 = torch.tanh(out1)
        out2, (h_n2, h_c2) = self.lstm_layer2(out1)
        h_n2 = self.drop(h_n2)
        a, b, c = h_n2.shape
        out3 = self.linear_layer(h_n2.reshape(a * b, c))
        return out3


# 呈现原始数据，训练结果，验证结果，预测结果
def plot_img(label, predict):
    plt.figure(figsize=(24, 8))
    # 原始数据蓝色
    plt.plot([x for x in label], c='b', label='label')
    # 训练数据绿色
    plt.plot([x for x in predict], c='g', label='predict')
    plt.legend()
    plt.savefig('out.png')
    plt.show()


def train(dataloader, model):
    epochs = 500
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        total_train_loss = 0
        total_train_num = 0
        for x, y in tqdm(dataloader,
                         desc='data'):
            x_num = len(x)
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            loss = loss_func(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_num += x_num
        train_loss = total_train_loss / total_train_num
        print("{}/epoch loss:".format(epoch), train_loss)
    torch.save(model.state_dict(), './lstm_stock.pth')


def test(dataloader, model):
    pred = []
    label = []
    model.load_state_dict(torch.load("./lstm_stock.pth",
                                     map_location=torch.device('cpu')))
    model.eval()
    for x, y in dataloader:
        x = x.to(device)
        p = model(x).cpu()
        pred.extend(p.data.squeeze(1).tolist())
        label.extend(y.tolist())
    return pred, label


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = MyDataset()
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=100,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            sampler=sampler)
    model = Lstm_Model().to(device)
    #  模型训练 #
    # train(dataloader, model)

    # 模型测试  #
    pred, label = test(dataloader, model)
    print(len(pred))
    print(len(label))
    plot_img(label, pred)
