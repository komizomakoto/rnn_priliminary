from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

N_INPUTS = 4
N_TIME = 10
THIN_PART = 1
THIN_PART2 = 20
PATICULAR_ID = 2

def create_dataset(N_TIME,N_INPUTS):
    data_num = 0
    train_data_num = 0
    input_data = np.zeros((1, N_TIME, N_INPUTS))
    correct_data = np.zeros((1, 1))

    for i in range(THIN_PART,30):
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
                input_data_ij[_] = np_input[_:_+N_TIME, 0:0+N_INPUTS]
            input_data = np.concatenate([input_data, input_data_ij])

            #### correct data (m, 1)
            df_correct = pd.read_csv(path_ij_tag, index_col=0)
            np_correct = df_correct.to_numpy()
            np_correct = np_correct[1:]
            correct_data_ij = np.zeros((n_sample_ij, 1))
            for _ in range(n_sample_ij):
                correct_data_ij[_] = np_correct[_]
            correct_data = np.concatenate([correct_data, correct_data_ij])

            #### related num
            train_data_num = train_data_num + input_data_ij.shape[0]
            data_num += 1
            # break

    # print("train_data_num", train_data_num)
    input_data = input_data[1:]
    input_data[:,:,0] = input_data[:,:,0]*0.01
    input_data[:,:,1] = input_data[:,:,1]*0.01
    correct_data = correct_data[1:]
    return input_data, correct_data

def main():
    input_data, correct_data = create_dataset(N_TIME,N_INPUTS)
    X_train, X_test, Y_train, y_test = train_test_split(input_data, correct_data, test_size = 0.2, random_state = 1)
    print("input_data shape, ", input_data.shape, "correct_data shape", correct_data.shape)
    print("X_train shape, ", X_train.shape, "y_train shape", X_train.shape)

    #### model structure
    out_neurons = 1
    n_hidden = 32

    model = Sequential()
    model.add(LSTM(n_hidden, batch_input_shape=(None, N_TIME, N_INPUTS), return_sequences=True))
    model.add(LSTM(n_hidden, batch_input_shape=(None, N_TIME, N_INPUTS), return_sequences=False))
    model.add(Dense(out_neurons))
    model.add(Activation("sigmoid"))
    optimizer = RMSprop(lr=0.1)
    model.compile(loss="binary_crossentropy", 
    optimizer=optimizer,
    metrics=['accuracy'])

    model.summary()

    #### model learning
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    history = model.fit(input_data, correct_data,
            batch_size=200,
            epochs=100,
            validation_split=0.01,
            callbacks=[early_stopping]
            )

    #### prediction
    predicted = model.predict(X_test)
    print("input_data shape, ", input_data.shape, "correct_data shape", correct_data.shape)
    print("num of that correct_data has 1 flag", correct_data[correct_data>0.5].shape)
    print("num of that predicted has 1 flag", predicted[predicted>0.5].shape)

    predicted[predicted>0.5] = 1
    predicted[predicted<=0.5] = 0
    cm = confusion_matrix(y_test, predicted)
    accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    precision = (cm[1,1]/(cm[0,1] + cm[1,1]))
    recall = (cm[1,1]/(cm[1,1] + cm[1,0]))
    specificity = (cm[0,0]/(cm[0,0] + cm[0,1]))
    F = (2*recall*precision/(recall+precision))
    print("cm", cm)
    print("Accuracy", accuracy)
    print("Precision", precision)
    print("Recall", recall)
    print("Specificity", specificity)
    print("F-measure", F)

    #### make figure
    plt.figure()
    # compare prediction with correct (divide train and test data by tensorflow)
    # plt.plot(range(0, correct_data.shape[0]), correct_data, color="r", label="row_data")
    # plt.plot(range(0, predicted.shape[0]), predicted, color="b", label="predict_data")

    # compare prediction with correct (divide train and test data by sklearn)
    plt.plot(range(0, y_test.shape[0]), y_test, color="r", label="row_data")
    plt.plot(range(0, predicted.shape[0]), predicted, color="b", label="predict_data")
    # plt.plot(range(0, 50), y_test[0:50], color="r", label="row_data")
    # plt.plot(range(0, 50), predicted[0:50], color="b", label="predict_data")
    plt.legend()

    # see train data merely
    # plt.plot(range(0, input_data.shape[0]), input_data[:,0,0], color="r", label="row_data")
    # plt.plot(range(0, input_data.shape[1]), input_data[0,:,0], color="r", label="row_data")
    # plt.plot(range(0, input_data.shape[1]), correct_data[0], color="r", label="row_data")

    # plot the model accuracy and validation accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.legend(['accuracy','validation accuracy'])
    plt.show()
    model.save('./save_model/my_model.h5')

if __name__ == "__main__":
    main()