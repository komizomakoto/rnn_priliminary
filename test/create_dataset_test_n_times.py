# PURPOSE: for test use, load ALL data(output_data/each_object/output_test_*_*.csv) and create dataset

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
N_INPUTS = 4

def create_dataset(N_TIME,N_INPUTS):
    data_num = 0
    train_data_num = 0
    input_data = np.zeros((1, N_TIME, N_INPUTS))
    correct_data = np.zeros((1, 1))

    for i in range(1,30):
        for j in range(1,11):
            path_ij = "output_data/each_object/output_test_{}_{}.csv".format(i,j)
            path_ij_tag = "output_data/annotation/annotation_{}_{}.csv".format(i,j)
            
            datalist = glob.glob(path_ij)
            if datalist == []:
                continue
            
            #### input data (m, 4)
            df_input = pd.read_csv(path_ij, index_col=0)
            df_input = df_input.drop(["time", 'id', 'class'], axis=1) 
            np_input = df_input.to_numpy()
            np_input = np_input[1:]
            if np_input.shape[0] < N_TIME:
                continue
            n_sample_ij = np_input.shape[0] - N_TIME + 1
            input_data_ij = np.zeros((n_sample_ij, N_TIME, N_INPUTS))
            for _ in range(n_sample_ij):
                input_data_ij[_] = np_input[_:_+N_TIME]
            # print("input_data_ij", input_data_ij.shape)

            input_data = np.concatenate([input_data, input_data_ij])

            #### correct data (m, 1)
            df_correct = pd.read_csv(path_ij_tag, index_col=0)
            np_correct = df_correct.to_numpy()
            np_correct = np_correct[1:]
            correct_data_ij = np.zeros((n_sample_ij, 1))
            for _ in range(n_sample_ij):
                correct_data_ij[_] = np_correct[_]
            # print("correct_data_ij", correct_data_ij.shape)

            correct_data = np.concatenate([correct_data, correct_data_ij])

            #### related num
            train_data_num = train_data_num + input_data_ij.shape[0]
            data_num += 1

    # print("train_data_num", train_data_num)
    input_data = input_data[1:]
    correct_data = correct_data[1:]
    return input_data, correct_data


input_data, correct_data = create_dataset(N_TIME,N_INPUTS)
print("input_data", input_data.shape)
print("input_data", input_data.sum())
print("correct_data", correct_data.shape)
print("correct_data", correct_data.sum())
