import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from time import time
import numpy as np
import random
import os
import glob
import matplotlib.pyplot as plt

EPOCH = 50                           # number of epochs
BATCH = 32
LR = 5e-2                            # learning rate of the gradient descent
LAMBD = 3e-2                         # lambda in L2 regularizaion
DP = 0.0                             # dropout rate
RDP = 0.0                            # recurrent dropout rate
TIMESETPS = 150                              # timesteps
LAYERS = [8, 8, 8, 1]                # number of units in hidden and output layers

def set_train_dataset():
    A_list = []
    csv_folder_path = 'data/bw_clip/'
    subfolders = ['parkin', 'normal']
    for subfolder in subfolders:
        csv_sub_folder_path = os.path.join(csv_folder_path, subfolder)
        csv_files = glob.glob(os.path.join(csv_sub_folder_path, "*.csv"))
        for file in csv_files:
            df = pd.read_csv(file)
            stride = 50    # ずらす行数
            X = []  # 入力データのリスト
            num_samples = (len(df) - TIMESETPS) // stride + 1
            for i in range(num_samples):
                start_row = i * stride
                end_row = start_row + TIMESETPS
                X = df.iloc[start_row:end_row].values
                y = 1 if subfolder == 'parkin' else 0
                A = {'X': X, 'y': y}  # AオブジェクトにXとyを格納
                A_list.append(A)            # Aオブジェクトをリストに追加

    print(f"number of sample : {len(A_list)}")
    random.shuffle(A_list)
    X_array_list = np.array([A['X'] for A in A_list])
    y_array_list = np.array([A['y'] for A in A_list])

    print(f"X shape: {X_array_list.shape}")
    print(f"y shape: {y_array_list.shape}")
    return X_array_list, y_array_list

def create_model(TIMESETPS, num_of_features):
    model = Sequential()
    model.add(LSTM(input_shape=(TIMESETPS, num_of_features), units=LAYERS[0],
                activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                dropout=DP, recurrent_dropout=RDP,
                return_sequences=True, return_state=False,
                stateful=False, unroll=False
                ))
    model.add(BatchNormalization())
    model.add(LSTM(units=LAYERS[1],
                activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                dropout=DP, recurrent_dropout=RDP,
                return_sequences=True, return_state=False,
                stateful=False, unroll=False
                ))
    model.add(BatchNormalization())
    model.add(LSTM(units=LAYERS[2],
                activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                dropout=DP, recurrent_dropout=RDP,
                return_sequences=False, return_state=False,
                stateful=False, unroll=False
                ))
    model.add(BatchNormalization())
    model.add(Dense(units=LAYERS[3], activation='sigmoid'))

    # Compile the model with Adam optimizer
    model.compile(loss='binary_crossentropy',
                metrics=['accuracy'],
                optimizer='adam')
    print(model.summary())
    return model

def evaluate_model(model, start, X_train, y_train, X_test, y_test, num_of_train, num_of_test):
    print('-'*65)
    print(f'Training was completed in {time() - start:.2f} secs')
    print('-'*65)
    # Evaluate the model:
    train_loss, train_acc = model.evaluate(X_train, y_train,
                                        batch_size=num_of_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test[:num_of_test], y_test[:num_of_test],
                                        batch_size=num_of_test, verbose=0)
    print('-'*65)
    print(f'train accuracy = {round(train_acc * 100, 4)}%')
    print(f'test accuracy = {round(test_acc * 100, 4)}%')
    print(f'test error = {round((1 - test_acc) * num_of_test)} out of {num_of_test} examples')


def print_history(History):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
    axs[0].plot(History.history['loss'], color='b', label='Training loss')
    axs[0].plot(History.history['val_loss'], color='r', label='Validation loss')
    axs[0].set_title("Loss curves")
    axs[0].legend(loc='best', shadow=True)
    axs[1].plot(History.history['accuracy'], color='b', label='Training accuracy')
    axs[1].plot(History.history['val_accuracy'], color='r', label='Validation accuracy')
    axs[1].set_title("Accuracy curves")
    axs[1].legend(loc='best', shadow=True)
    plt.show()



def main():
    X_array_list, y_array_list = set_train_dataset()
    X_train, X_temp, y_train, y_temp = train_test_split(X_array_list, y_array_list, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f'Train data dimensions: {X_train.shape}, {y_train.shape}')
    print(f'Test data dimensions: {X_test.shape}, {y_test.shape}')
    print(f'Validation data dimensions: {X_val.shape}, {y_val.shape}')

    num_of_train = X_train.shape[0]           # number of training examples (2D)
    num_of_test = X_test.shape[0]
    num_of_features = X_array_list.shape[2]                 # number of features
    batch_size = num_of_train                          # batch size

    print(f'layers={LAYERS}, train_examples={num_of_train}, test_examples={num_of_test}')
    print(f'batch = {batch_size}, timesteps = {TIMESETPS}, features = {num_of_features}, epochs = {EPOCH}')
    print(f'lr = {LR}, lambda = {LAMBD}, dropout = {DP}, recurr_dropout = {RDP}')

    lr_decay = ReduceLROnPlateau(monitor='loss', 
                                patience=1, verbose=0, 
                                factor=0.5, min_lr=1e-8)
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, 
                            patience=30, verbose=1, mode='auto',
                            baseline=0, restore_best_weights=True)

    start = time()
    model = create_model(TIMESETPS, num_of_features)
    History = model.fit(X_train, y_train,
                        epochs=EPOCH,
                        batch_size=BATCH,
                        validation_data=(X_val, y_val),
                        shuffle=True,verbose=0,
                        callbacks=[lr_decay, early_stop])

    evaluate_model(model, start, X_train, y_train, X_test, y_test, num_of_train, num_of_test)
    print_history(History)


if __name__ == "__main__":
    main()