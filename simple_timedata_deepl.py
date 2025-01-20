
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from scikeras.wrappers import KerasClassifier
from keras.regularizers import l2
from time import time
import numpy as np
import csv
import random
import os
import glob
import matplotlib.pyplot as plt
from collections import defaultdict


EPOCH = 50                           # number of epochs
BATCH = 32
LR = 5e-2                            # learning rate of the gradient descent
LAMBD = 0                       # lambda in L2 regularizaion
DP = 0.0                             # dropout rate
RDP = 0.0                            # recurrent dropout rate
TIMESETPS = 150                              # timesteps
LAYERS = [64, 64, 64, 1]                # number of units in hidden and output layers
# STRIDE = int(TIMESETPS / 2)
STRIDE = 50
NUM_FEATURES = 3

class SampleData:
    def __init__(self, train_array, target_value, file_name):
        self.file_name = file_name
        self.train_array = train_array
        self.target_value = target_value

class ResultData:
    def __init__(self, predict_value, file_name):
        self.predict_value = predict_value
        self.file_name = file_name

# カスタムコールバッククラス
class HistoryLogger(Callback):
    def __init__(self):
        super(HistoryLogger, self).__init__()
        self.history = None

    def on_train_begin(self, logs=None):
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        for key in logs.keys():
            self.history[key].append(logs[key])


def split_val_csv_files(csv_sub_folder_path, subfolder):
    if subfolder == "parkin":
        x = np.random.randint(23, 103, 22)
        filename_tamplate = "tone{0}_neck{1}_clipout.csv"
    else:
        x = np.random.randint(1, 33, 7)
        filename_tamplate = "{0}_neck{1}_clipout.csv"

    # ここで、テスト用のデータ20%を取り除く
    selected_files = []
    for xdayo in x:
        xdayo = str(xdayo).zfill(3)
        for num in [1,2]:
            selected_file_name = os.path.join(csv_sub_folder_path, filename_tamplate.format(xdayo, num))
            selected_files.append(selected_file_name)
            
    csv_files = glob.glob(os.path.join(csv_sub_folder_path, "*.csv"))
    remaining_files = [f for f in csv_files if f not in selected_files]
    return remaining_files, selected_files

def make_sample_data(files, subfolder):
    sample_data_list = []
    for file in files:
        try:
            df = pd.read_csv(file)
            X = []  # 入力データのリスト
            stride = 50
            num_samples = (len(df) - TIMESETPS) // STRIDE + 1
            for i in range(num_samples):
                start_row = i * STRIDE
                end_row = start_row + TIMESETPS
                X = df.iloc[start_row:end_row].values
                y = 1 if subfolder == 'parkin' else 0
                file_name = os.path.splitext(os.path.basename(file))[0]
                sample_data = SampleData(X, y, file_name)
                sample_data_list.append(sample_data)        # Aオブジェクトをリストに追加
        except FileNotFoundError:
            print(f"File {file} not found, skipping.")
    return sample_data_list

    

def set_train_dataset():
    train_list = []
    test_list = []
    csv_folder_path = 'data/bw_clip/'
    subfolders = ['parkin', 'normal']
    for subfolder in subfolders:
        csv_sub_folder_path = os.path.join(csv_folder_path, subfolder)
        remaining_files, selected_files = split_val_csv_files(csv_sub_folder_path, subfolder)
        train_list += make_sample_data(remaining_files, subfolder)
        test_list += make_sample_data(selected_files, subfolder)

    print(f"number of sample : {len(train_list)}")
    random.shuffle(train_list)
    X_train_list = np.array([sample.train_array for sample in train_list])
    y_train_list = np.array([sample.target_value for sample in train_list])
    print(f"X_train_list shape: {X_train_list.shape}")
    print(f"y_train_list shape: {y_train_list.shape}")

    random.shuffle(test_list)
    X_test_list = np.array([sample.train_array for sample in test_list])
    y_test_list = np.array([sample.target_value for sample in test_list])

    print(f"X_test_list shape: {X_test_list.shape}")
    print(f"y_test_list shape: {y_test_list.shape}")

    return X_train_list, y_train_list, X_test_list, y_test_list, test_list

def create_model(layer_units, activation, dropout_rate=0.0, recurrent_dropout_rate=0.0):
    model = Sequential()
    for i, units in enumerate(layer_units):
        return_sequences = (i != len(layer_units) - 1)
        model.add(LSTM(units=units, activation=activation, recurrent_activation='hard_sigmoid',
                       kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                       dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                       return_sequences=return_sequences, input_shape=(TIMESETPS, NUM_FEATURES)))
        model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, start, X_train, y_train, X_test, y_test, num_of_test):
    print('-'*65)
    print(f'Training was completed in {time() - start:.2f} secs')
    print('-'*65)
    # Evaluate the model:
    # train_loss, train_acc = model.evaluate(X_train, y_train,
    #                                     batch_size=num_of_train, verbose=0)
    # test_loss, test_acc = model.evaluate(X_test[:num_of_test], y_test[:num_of_test],
    #                                     batch_size=num_of_test, verbose=0)
    train_acc = model.score(X_train, y_train)
    test_acc =model.score(X_test, y_test)
    
    print('-'*65)
    print(f'train accuracy = {round(train_acc * 100, 4)}%')
    print(f'test accuracy = {round(test_acc * 100, 4)}%')
    print(f'test error = {round((1 - test_acc) * num_of_test)} out of {num_of_test} examples')


def print_history(history):
    val_accuracy = history['val_accuracy']
    print("Validation Accuracy over epochs:", val_accuracy)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
    axs[0].plot(history['loss'], color='b', label='Training loss')
    axs[0].plot(history['val_loss'], color='r', label='Validation loss')
    axs[0].set_title("Loss curves")
    axs[0].legend(loc='best', shadow=True)
    axs[1].plot(history['accuracy'], color='b', label='Training accuracy')
    axs[1].plot(history['val_accuracy'], color='r', label='Validation accuracy')
    axs[1].set_title("Accuracy curves")
    axs[1].legend(loc='best', shadow=True)
    plt.show()


# CSVファイルに結果を書き込む関数
def write_results_to_csv(result_list, output_file):
    result_dict = {}
    for result in result_list:
        if result.file_name in result_dict:
            result_dict[result.file_name].append(result.predict_value)
        else:
            result_dict[result.file_name] = [result.predict_value]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        max_columns = max(len(values) for values in result_dict.values())
        header = ['file_name'] + [f'predict_value_{i+1}' for i in range(max_columns)]
        writer.writerow(header)
        
        # データの書き込み
        for file_name, predict_values in result_dict.items():
            row = [file_name] + predict_values
            writer.writerow(row)

def judge_by_majority_form_csv():
    df = pd.read_csv('output.csv')
    atari = 0

    # 行ごとに読み取り
    for index, row in df.iterrows():
        isParkin = True if 'tone' in row['file_name'] else False
        judge, sum_value, parkin_label, normal_label = 0, 0, 0,0

        # 列ごとに読み取り
        for column in df.columns[1:]: 
            if column not in row or pd.isna(row[column]):
                break  
            value = row[column]
            sum_value += value
            if value >= 0.5:
                parkin_label+=1
            else:
                normal_label+=1
            
            judge = parkin_label - normal_label
        average = sum_value / len(df.columns[1:])

        # 推定が当たりかどうかの判断
        if (isParkin and judge > 0) or (not isParkin and judge < 0):
            atari += 1
        elif judge == 0:
            if (isParkin and average >= 0.5) or (isParkin and average < 0.5):
                atari += 1

        print(f"{row['file_name']}: ({parkin_label}, {normal_label})")

    acc = atari * 100 / len(df)
    print(f"正解率: {acc}%")


def main():
    X_train, y_train, X_test, y_test, test_data_list = set_train_dataset()
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    print(f'Train data dimensions: {X_train.shape}, {y_train.shape}')
    print(f'Test data dimensions: {X_test.shape}, {y_test.shape}')
    # print(f'Validation data dimensions: {X_val.shape}, {y_val.shape}')

    num_of_train = X_train.shape[0]           # number of training examples (2D)
    num_of_test = X_test.shape[0]
    num_of_features = X_train.shape[2]                 # number of features

    print(f'layers={LAYERS}, train_examples={num_of_train}, test_examples={num_of_test}')
    print(f'batch = {BATCH}, timesteps = {TIMESETPS}, features = {num_of_features}, epochs = {EPOCH}')
    print(f'lr = {LR}, lambda = {LAMBD}, dropout = {DP}, recurr_dropout = {RDP}')

    lr_decay = ReduceLROnPlateau(monitor='loss', 
                                patience=1, verbose=1, 
                                factor=0.5, min_lr=1e-8)
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, 
                            patience=10, verbose=1, mode='auto',
                            baseline=0, restore_best_weights=True)

    start = time()
    param_grid = {
        'batch_size': [32, 64, 256],
        'epochs': [50],
        'layer_units': [[8, 8], [64, 64], [8,8,8], [64, 64, 64]],
        'activation' : ['tanh', 'relu']
        # 'dropout_rate': [0.0, 0.2],
        # 'recurrent_dropout_rate': [0.0, 0.2],
    }
    model = KerasClassifier(build_fn=create_model, layer_units = [8,8], activation = 'tanh')
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_result = grid.fit(
        X_train, y_train, callbacks=[lr_decay, early_stop], shuffle=True, verbose=1, epochs=EPOCH, batch_size=BATCH)
    best_model = grid_result.best_estimator_

    # # model = create_model(TIMESETPS, num_of_features)
    # History = best_model.fit(X_train, y_train,
    #                     epochs=EPOCH,
    #                     batch_size=BATCH,
    #                     validation_data=(X_test, y_test),
    #                     shuffle=True,verbose=0,
    #                     callbacks=[lr_decay, early_stop])

    predictions = best_model.predict(X_test)
    result_list = []
    for i, predict in enumerate(predictions):
        sample_data = test_data_list[i]
        result = ResultData(predict, sample_data.file_name)
        result_list.append(result)

    write_results_to_csv(result_list, 'output.csv')
    evaluate_model(best_model, start, X_train, y_train, X_test, y_test, num_of_test)
    # print_history(best_model)


if __name__ == "__main__":
    main()
    judge_by_majority_form_csv()
