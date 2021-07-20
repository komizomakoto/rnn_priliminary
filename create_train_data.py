# PURPOSE: create train data from each object list 
# DESCRIPTION: from "output_data/each_object/output_test_1_1.csv"

import pandas as pd
import numpy as np
#### DATA RESHAPE ####
import torch
from torch.utils.data import DataLoader


file_num = 1
i = 1
anno_num = 1
anno_car_id = 1
df_input = pd.read_csv("output_data/each_object/output_test_{}_{}.csv".format(file_num,i), index_col=0)
df_input = df_input.drop(["time", "id", 'class'], axis=1) 
# print("df_input", df_input)
nmp_input=df_input.to_numpy()
# print("nmp_input", nmp_input.shape)
df_correct = pd.read_csv("output_data/annotation/annotation_{}_{}.csv".format(anno_num,anno_car_id), index_col=0)
nmp_correct=df_correct.to_numpy()

n_time = 10 # 時系列の数
n_sample = len(nmp_input)-n_time # サンプル数
print("n_sample", n_sample)

input_data = np.zeros((n_sample, n_time, 4)) # 入力
correct_data = np.zeros((n_sample, 1)) # 正解

for i in range(n_sample):
    input_data[i] = nmp_input[i:i+n_time].reshape(-1, 4)
    correct_data[i] = nmp_correct[i+n_time:i+n_time+1] # 正解は入力よりも一つ後

input_data = torch.tensor(input_data, dtype=torch.float) # 
correct_data = torch.tensor(correct_data, dtype=torch.float)
dataset = torch.utils.data.TensorDataset(input_data, correct_data) # 
# print("input_data", input_data)
# print("correct_data", correct_data)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True) # DataLoaderの設定

#### MODEL CREATE ####
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device:', device)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=4,hidden_size=64,batch_first=True)
        # RNN層,入力サイズ,ニューロン数,入力を (バッチサイズ, 時系列の数, 入力の数) にする
        self.fc = nn.Linear(64, 1) # 全結合層
    def forward(self, x):
        print("x", x)

        y_rnn, h = self.rnn(x, None) # hは次の時刻に渡される値、 Noneでその初期値が0に
        print("y_rnn", y_rnn)
        print("h", h)

        y = self.fc(y_rnn[:, -1, :]) # yは最後の時刻の出力
        print("###")
        return y

net = Net()
print(net)

#### LEARN ####
from torch import optim

# 交差エントロピー誤差関数
loss_fnc = nn.MSELoss()

# 最適化アルゴリズム
optimizer = optim.SGD(net.parameters(), lr=0.01) # 学習率は0.01

# 損失のログ
record_loss_train = []
# net = net.to(device)
# 学習
for i in range(50): # 50エポック学習
    net.train() # 訓練モード
    loss_train = 0
    for j, (x, t) in enumerate(train_loader): # ミニバッチ（x, t）を取り出す
        y = net(x)
        # print("x", x)
        # print("t", t)
        loss = loss_fnc(y, t)
        # print("loss", loss)

        loss_train += loss.item()
        # print("loss_train", loss_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # loss_train /= j+1
    # record_loss_train.append(loss_train)
    # print("#########################")

    # if i%2 == 0:
    #     print("Epoch:", i, "Loss_Train:", loss_train)
    #     predicted = list(input_data[0].reshape(-1)) # 最初の入力
    #     for i in range(n_sample):
    #         x = torch.tensor(predicted[-n_time:]) # 直近の時系列を取り出す
    #         x = x.reshape(1, n_time, 1) # (バッチサイズ, 時系列の数, 入力の数)
    #         y = net(x)
    #         predicted.append(y[0].item()) # 予測結果をpredictedに追加する

    # plt.plot(range(len(sin_y)), sin_y, label="Correct")
    # plt.plot(range(len(predicted)), predicted, label="Predicted")
    # plt.legend()
    # plt.show()




# df_input_converted_tmp = 


# columns = ['time', 'class', 'region1x', 'region1y', 'region2x', 'region2y']

# file_num = 0

# while True:
#     file_num += 1
#     try:
#         # print("here",pd.read_csv('./output_data/output{}.csv'.format(1), index_col=0) )
#         df = pd.read_csv('./output_data/mot_out/output{}.csv'.format(file_num), index_col=0)
#         df["centerx"] = (df["region1x"] + df["region2x"])/2 
#         df["centery"] = (df["region1y"] + df["region2y"])/2
#         df = df.drop(["region1x", 'region1y', 'region2x', 'region2y'], axis=1) 

#         # i=1
#         # while True:
#         for i in range(1,11):
#             if df[df['id']==i].empty:
#                 continue
#             df["diff_x"] = df[df['id']==i]["centerx"].diff()
#             df["diff_y"] = df[df['id']==i]["centery"].diff()
#             # print("########")
#             # print(df[df['id']==i])
#             df[df['id']==i].to_csv("output_data/each_object/output_test_{}_{}.csv".format(file_num,i))
#             # i += 1
#         print('file_num i', file_num, i)
#     except:
#         break


