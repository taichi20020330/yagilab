import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

# データの準備
X = np.random.rand(2000, 100)  # 2000サンプル、100次元
y = np.random.randint(0, 2, 2000)  # 0か1の二値ラベル

# データをトレーニング、バリデーション、テストに分割 なんかこの分割違うらしい
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# モデルの構築
model = Sequential([
    Dense(64, input_dim=100, activation='relu'),  # 第1層
    Dense(32, activation='relu'),                # 第2層
    Dense(1, activation='sigmoid')               # 出力層
])

# モデルのコンパイル
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# モデルの訓練とバリデーション
history = model.fit(X_train, y_train, epochs=20, batch_size=32, 
                    validation_data=(X_val, y_val))

# テストデータでの評価
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# 学習曲線のプロット（任意）
import matplotlib.pyplot as plt

# 訓練とバリデーションの損失をプロット
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 訓練とバリデーションの精度をプロット
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
