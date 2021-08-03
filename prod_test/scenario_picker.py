from tensorflow.keras import models
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

def main():
    new_model = models.load_model('./save_model/my_model.h5')
    new_model.summary()

    #### prediction
    X_test = 
    predicted = model.predict(X_test)
    print("input_data shape, ", input_data.shape, "correct_data shape", correct_data.shape)
    print("num of that correct_data has 1 flag", correct_data[correct_data>0.5].shape)
    print("num of that predicted has 1 flag", predicted[predicted>0.5].shape)

    predicted[predicted>0.5] = 1
    predicted[predicted<=0.5] = 0
    # # cm = confusion_matrix(y_test, predicted)
    # # accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    # # precision = (cm[1,1]/(cm[0,1] + cm[1,1]))
    # # recall = (cm[1,1]/(cm[1,1] + cm[1,0]))
    # # specificity = (cm[0,0]/(cm[0,0] + cm[0,1]))
    # # F = (2*recall*precision/(recall+precision))
    # print("cm", cm)
    # print("Accuracy", accuracy)
    # print("Precision", precision)
    # print("Recall", recall)
    # print("Specificity", specificity)
    # print("F-measure", F)


if __name__ == "__main__":
    main()