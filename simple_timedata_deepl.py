import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob

sample_num = 10
size = 3000
dimention = 3
epochs = 20
batch_size = 32
A_list = []

# X: 10こ、3000行、3つの統計量→ 3000 * 3の配列が10こ   ...10 * [3000, 3]
# Y: 10このスカラ値

X = np.random.rand(sample_num, size, dimention) 
y = np.random.randint(0, 2, sample_num) 

csv_folder_path = 'data/bw_clip/'
subfolders = ['parkin', 'normal']
for subfolder in subfolders:
    csv_sub_folder_path = os.path.join(csv_folder_path, subfolder)
    csv_files = glob.glob(os.path.join(csv_sub_folder_path, "*.csv"))
    for file in csv_files:
        df = pd.read_csv(file)
        print(file)  # 最初の5行を表示

        n_steps = 100  # タイムステップの数
        stride = 50    # ずらす行数
        feature_columns = df.columns  # すべての列を特徴量として使用
        X = []  # 入力データのリスト
        num_samples = (len(df) - n_steps) // stride + 1

        for i in range(num_samples):
            start_row = i * stride
            end_row = start_row + n_steps
            X = df.iloc[start_row:end_row].values
            y = 1 if subfolder == 'parkin' else 0
            A = {'X': X, 'y': y}  # AオブジェクトにXとyを格納
            A_list.append(A)            # Aオブジェクトをリストに追加

    print(len(A_list))


# A_index = 5  # 5番目のAオブジェクトにアクセス（0インデックス）
# X_value = A_list[A_index]['X']

# # 確認のためにXの形状を表示
# print(f"4番目のAオブジェクトのXの値:\n{A_list[170]['X']}")
# print(f"4番目のAオブジェクトのyの値:\n{A_list[170]['y']}")
# print(f"9番目のAオブジェクトのXの値:\n{A_list[9]['X']}")
# print(f"9番目のAオブジェクトのyの値:\n{A_list[9]['y']}")

X_array_list = np.array([A['X'] for A in A_list])
y_array_list = np.array([A['y'] for A in A_list])



# # # データをトレーニング、バリデーション、テストに分割
X_train, X_temp, y_train, y_temp = train_test_split(X_array_list, y_array_list, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# モデルの構築
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(size, dimention)),  # LSTM層
    Dense(1, activation='sigmoid')                         # 出力層
])

# モデルのコンパイル
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# モデルの訓練とバリデーション
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_val, y_val), callbacks=[early_stopping])

# テストデータでの評価
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# 学習曲線のプロット（任意）
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
