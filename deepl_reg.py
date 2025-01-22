
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
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


EPOCH = 30                           # number of epochs
BATCH = 50
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



def ccc(y_true, y_pred):
    """Concordance Correlation Coefficient (CCC)"""
    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)
    
    cov = tf.reduce_mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    y_true_var = tf.reduce_mean(tf.square(y_true - y_true_mean))
    y_pred_var = tf.reduce_mean(tf.square(y_pred - y_pred_mean))
    
    ccc_val = (2 * cov) / (y_true_var + y_pred_var + tf.square(y_true_mean - y_pred_mean))
    return ccc_val

def r2_score(y_true, y_pred):
    """R-squared (R²)"""
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())
    return r2


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
    updrs_df = pd.read_csv("features/updrs_list_2023.csv")
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
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=LR), 
              loss='mean_squared_error', 
              metrics=['mae', ccc, r2_score])    
    return model


def evaluate_model(model, start, X_train, y_train, X_test, y_test, num_of_test):
    print('-'*65)
    print(f'Training was completed in {time() - start:.2f} secs')
    print('-'*65)
    
    train_results = model.evaluate(X_train, y_train, batch_size=len(X_train), verbose=0)
    test_results = model.evaluate(X_test[:num_of_test], y_test[:num_of_test], batch_size=num_of_test, verbose=0)
    
    print('-'*65)
    print(f'Train Loss = {train_results[0]:.4f}, Train MAE = {train_results[1]:.4f}, Train CCC = {train_results[2]:.4f}, Train R² = {train_results[3]:.4f}')
    print(f'Test Loss = {test_results[0]:.4f}, Test MAE = {test_results[1]:.4f}, Test CCC = {test_results[2]:.4f}, Test R² = {test_results[3]:.4f}')
    print('-'*65)



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
        'batch_size': [64, 256],
        # 'layer_units': [[8, 8], [64, 64], [8,8,8], [64, 64, 64]],
        # 'activation' : ['tanh', 'relu'],
        # 'dropout_rate': [0.0, 0.2],
        # 'recurrent_dropout_rate': [0.0, 0.2],
    }
    model = KerasClassifier(model=create_model, layer_units=[8, 8], activation='tanh', verbose=1)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
    grid_result = grid.fit(
        X_train, y_train, callbacks=[lr_decay, early_stop], verbose=1)
    # 最適なパラメータの取得
    best_params = grid_result.best_params_
    print("Best parameters found:", best_params)
    best_model = grid_result.best_estimator_

    # predictions = best_model.predict(X_test)
    # result_list = []
    # for i, predict in enumerate(predictions):
    #     sample_data = test_data_list[i]
    #     result = ResultData(predict, sample_data.file_name)
    #     result_list.append(result)

    # write_results_to_csv(result_list, 'output.csv')
    # evaluate_model(best_model, start, X_train, y_train, X_test, y_test, num_of_test)
    # print_history(best_model)


if __name__ == "__main__":
    main()
    judge_by_majority_form_csv()



    # # model = create_model(TIMESETPS, num_of_features)
    # History = best_model.fit(X_train, y_train,
    #                     epochs=EPOCH,
    #                     batch_size=BATCH,
    #                     validation_data=(X_test, y_test),
    #                     shuffle=True,verbose=0,
    #                     callbacks=[lr_decay, early_stop])