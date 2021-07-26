# PURPOSE: for test use, load data(output_data/each_object/output_test_1_1.csv) only once

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import glob


N_TIME = 10
N_INPUT = 4
n_sample = 1024

datalist_tmp = glob.glob("output_data/each_object/output_test_1_1.csv")
# #### input data (m, 4)
df_input = pd.read_csv("output_data/each_object/output_test_1_1.csv", index_col=0)
df_input = df_input.drop(["time", 'id', 'class'], axis=1) 
np_input = df_input.to_numpy()
np_input = np_input[1:]
n_sample_1_1 = np_input.shape[0] - N_TIME + 1
input_data_1_1 = np.zeros((n_sample_1_1, N_TIME, N_INPUT))

# print("np_input", np_input[0:0+N_TIME].shape)
# print("np_input", np_input.shape)
# print("input_data_1_1", input_data_1_1)
print("input_data_1_1", input_data_1_1.shape)
# #### correct data (m, 1)
df_correct = pd.read_csv("output_data/annotation/annotation_1_1.csv", index_col=0)
df_correct = pd.read_csv("output_data/annotation/annotation_1_1.csv", index_col=0)
np_correct = df_correct.to_numpy()
np_correct = np_correct[1:]
correct_data_1_1 = np.zeros((n_sample_1_1, 1))

for i in range(n_sample_1_1):
    input_data_1_1[i] = np_input[i:i+N_TIME]
    correct_data_1_1[i] = np_correct[i:i+1]

print("correct_data_1_1", correct_data_1_1.shape)

test= np.zeros((n_sample_1_1, N_TIME, N_INPUT))
test2 = np.concatenate([test, input_data_1_1])
print("test", test2.shape)

# n_sample = 1024
# n_time = 10
# input_data = np.zeros((n_sample, n_time, 1)) # 入力
# correct_data = np.zeros((n_sample, 1)) # 正解
# print("input_data", input_data.shape)
# print("correct_data", correct_data.shape)
# for i in range(n_sample):
#     input_data[i] = sin_y[i:i+n_time].reshape(-1, 1)
#     correct_data[i] = sin_y[i+n_time:i+n_time+1] # 正解は入力よりも一つ後

