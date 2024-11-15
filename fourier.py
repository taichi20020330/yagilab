import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def fourier(data_series):
    # サンプリングレート（データの取得間隔が0.01秒ならサンプリングレートは100Hz）
    sampling_rate = 100  # Hz
    n = len(data_series)  # データの長さ

    data_centered = data_series - data_series[0]
    # フーリエ変換を実行
    fft_values = fft(data_centered)
    frequencies = fftfreq(n, d=1/sampling_rate)

    # 振幅スペクトルの計算（絶対値を取って正規化）
    amplitudes = np.abs(fft_values)

    # 周波数スペクトルのプロット
    plt.plot(frequencies[1:n // 2], amplitudes[1:n // 2])  # 正の周波数成分のみ表示
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum")
    plt.show()

    # 0 Hz（DC成分）を除外してピーク周波数を特定
    peak_frequency = frequencies[1 + np.argmax(amplitudes[1:n // 2])]

    period = 1 / peak_frequency  # 周期 (秒)

    # print(peak_frequency)


    print(f"主な周期: {period:.2f} 秒")



data_path = "./data/two_neck_dataset/normal/001_neck1_clipout.csv"
data = pd.read_csv(data_path, header=None, names=['x', 'y', 'z'])
fourier(data['z'])

