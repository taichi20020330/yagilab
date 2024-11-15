import numpy as np
import pandas as pd
from scipy import signal

def extract_features(time_series_data, fs=100):
    # データを2つの部分に分ける（例: 前半と後半）
    n = len(time_series_data)
    y1 = time_series_data[:n//2]
    y2 = time_series_data[n//2:]

    # デトレンド処理
    y1_detrended = signal.detrend(y1)
    y2_detrended = signal.detrend(y2)

    # Welchの方法でパワースペクトル密度を計算
    freq1, P1 = signal.welch(y1_detrended, fs, nperseg=512)
    freq2, P2 = signal.welch(y2_detrended, fs, nperseg=512)

    # パワースペクトルの平均
    P_mean = np.mean([P1, P2], axis=0)

    # Freezing Index (FI) の計算
    locomotor_band_power = np.sum(P_mean[(freq1 >= 0.5) & (freq1 < 3)])
    freezing_band_power = np.sum(P_mean[(freq1 >= 3) & (freq1 < 8)])
    freezing_index = freezing_band_power / locomotor_band_power if locomotor_band_power != 0 else 0

    # Central Frequency (CF) の計算
    weighted_frequency = np.sum(freq1 * (P_mean / np.sum(P_mean)))

    # Dominant Frequency (DF) の計算
    peaks, _ = signal.find_peaks(P_mean)
    dominant_frequency = freq1[peaks[np.argmax(P_mean[peaks])]] if peaks.size > 0 else 0

    # 振幅の計算
    amplitude = np.max(P_mean)

    # 相対的な振幅の計算
    relative_amplitude = amplitude / np.sum(P_mean) if np.sum(P_mean) != 0 else 0

    return {
        'Freezing Index': freezing_index,
        'Central Frequency': weighted_frequency,
        'Dominant Frequency': dominant_frequency,
        'Amplitude': amplitude,
        'Relative Amplitude': relative_amplitude
    }

# 使用例
time_series_data = np.random.randn(1000)  # 例としてランダムな時系列データ
features = extract_features(time_series_data)
print(features)
