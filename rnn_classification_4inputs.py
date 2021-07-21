#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

random.seed(777)
np.random.seed(777)
tf.set_random_seed(777)

# パラメーター
N_CLASSES = 1  # クラス数
N_INPUTS = 4  # 1ステップに入力されるデータ数
N_STEPS = 200  # 学習ステップ数
LEN_SEQ = 10  # 系列長
N_NODES = 64  # ノード数
N_DATA = 1024  # 各クラスの学習用データ数
N_TEST = 1000  # テスト用データ数
BATCH_SIZE = 20  # バッチサイズ

# データの準備

import pandas as pd
import numpy as np
#### DATA RESHAPE ####
import torch
from torch.utils.data import DataLoader

def load_data(file_num, car_id, anno_num, anno_car_id):
    df_input = pd.read_csv("output_data/each_object/output_test_{}_{}.csv".format(file_num,car_id), index_col=0)
    df_input = df_input.drop(["time", "id", 'class'], axis=1) 
    # print("df_input", df_input)
    nmp_input=df_input.to_numpy()
    # print("nmp_input", nmp_input.shape)
    df_correct = pd.read_csv("output_data/annotation/annotation_{}_{}.csv".format(anno_num,anno_car_id), index_col=0)
    nmp_correct=df_correct.to_numpy()

    n_time = LEN_SEQ # 時系列の数
    n_sample = len(nmp_input)-n_time # サンプル数
    # print("n_sample", n_sample)

    input_data = np.zeros((n_sample, n_time, N_INPUTS)) # 入力
    correct_data = np.zeros((n_sample, 1)) # 正解

    for i in range(n_sample):
        input_data[i] = nmp_input[i:i+n_time].reshape(-1, N_INPUTS)
        correct_data[i] = nmp_correct[i+n_time:i+n_time+1] # 正解は入力よりも一つ後
    
    for _ in range(9):
        input_data = np.concatenate([input_data, input_data])
        correct_data = np.concatenate([correct_data, correct_data])
    print("input_data", input_data.shape)
    print("correct_data", correct_data.shape)

    return input_data, correct_data

file_num = 1
car_id = 1
anno_num = 1
anno_car_id = 1
x_train, t_train = load_data(file_num, car_id, anno_num, anno_car_id)

# モデルの構築
x = tf.compat.v1.placeholder(tf.float32, [None, LEN_SEQ, N_INPUTS])  # 入力データ
t = tf.compat.v1.placeholder(tf.int32, [None])  # 教師データ

t_on_hot = tf.one_hot(t, depth=N_CLASSES, dtype=tf.float32)  # 1-of-Kベクトル

cell = rnn.BasicRNNCell(num_units=N_NODES, activation=tf.nn.tanh)  # 中間層のセル
# RNNに入力およびセル設定する
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, time_major=False)
# [ミニバッチサイズ,系列長,出力数]→[系列長,ミニバッチサイズ,出力数]
outputs = tf.transpose(outputs, perm=[1, 0, 2])

w = tf.Variable(tf.random_normal([N_NODES, N_CLASSES], stddev=0.01))
b = tf.Variable(tf.zeros([N_CLASSES]))
logits = tf.matmul(outputs[-1], w) + b  # 出力層
pred = tf.nn.softmax(logits)  # ソフトマックス

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t_on_hot, logits=logits)
loss = tf.reduce_mean(cross_entropy)  # 誤差関数
train_step = tf.train.AdamOptimizer().minimize(loss)  # 学習アルゴリズム

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(t_on_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 精度
print("pred",pred)

# 学習の実行
sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0
for _ in range(N_STEPS):
    cycle = int(N_DATA*3 / BATCH_SIZE)
    begin = int(BATCH_SIZE * (i % cycle))
    end = begin + BATCH_SIZE
    x_batch, t_batch = x_train[begin:end], t_train[begin:end]
    # print('x_batch', x_batch)
    print('t_batch', t_batch.shape)
    print('#############')
    sess.run(train_step, feed_dict={x:x_batch, t:t_batch})
    i += 1

    if i % 10 == 0:
        loss_, acc_ = sess.run([loss, accuracy], feed_dict={x:x_batch,t:t_batch})
        loss_test_, acc_test_ = sess.run([loss, accuracy], feed_dict={x:x_test,t:t_test})
        print("[TRAIN] loss : %f, accuracy : %f" %(loss_, acc_))
        print("[TEST loss : %f, accuracy : %f" %(loss_test_, acc_test_))
sess.close()
