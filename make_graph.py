import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import glob
import ruptures as rpt
from ruptures.metrics import precision_recall
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


base_input_folder = 'data/two_neck_dataset/'
base_output_folder = 'graph'
subfolders = ['parkin', 'normal']
output_folder_name =  'graph'
input_file_name = '*.csv'
statistic = ['x_acc', 'y_acc', 'z_acc']

import numpy as np
import matplotlib.pyplot as plt

def analyze_frequency_spectrum(df,sampling_rate=100):
    # z軸加速度データの取得
    acceleration_data = df['z_acc'].values - df['z_acc'].loc[0]
    
    # サンプル数
    n_samples = len(acceleration_data)
    
    # フーリエ変換
    fft_result = np.fft.fft(acceleration_data)
    
    # 周波数成分の強度を計算（絶対値）
    fft_magnitude = np.abs(fft_result)
    
    # 周波数軸の計算
    freqs = np.fft.fftfreq(n_samples, d=1/sampling_rate)
    
    # 正の周波数成分のみを取得
    positive_freqs = freqs[1:n_samples // 2]
    positive_magnitude = fft_magnitude[1:n_samples // 2]
    
    # 極大値（ピーク）を検出
    peaks, _ = find_peaks(positive_magnitude, height=np.max(positive_magnitude) * 0.3)  # 高さの閾値を調整可能
    
    # グラフのプロット
    plt.figure(figsize=(12, 6))
    plt.plot(positive_freqs, positive_magnitude, color='blue', label="Frequency Spectrum")
    plt.scatter(positive_freqs[peaks], positive_magnitude[peaks], color='red', label="Peaks", zorder=5)
    for peak in peaks:
        plt.text(positive_freqs[peak], positive_magnitude[peak], f"{positive_freqs[peak]:.2f} Hz", 
                 color="red", fontsize=10, ha="center", va="bottom")
    plt.title("Frequency Spectrum of Z-Acceleration with Peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def extract_local_maxima(signal_v):
    # 基準値を取得
    reference_value = signal_v.iloc[0]
    # 基準値との差を計算し、絶対値化
    distance_from_reference = (signal_v - reference_value).abs()
    
    # 時間と絶対値をデータフレーム化
    abs_dict = dict(time=signal_v.index, value=distance_from_reference.values)
    abs_df = pd.DataFrame(abs_dict)
    
    # # 極大値を取得
    # peak_indices, _ = find_peaks(abs_df['value'])
    # peak_times = abs_df['time'].iloc[peak_indices].values
    # peak_values = abs_df['value'].iloc[peak_indices].values

    # # 曲線補間 (スプライン補間)
    # interpolation_func = interp1d(
    #     peak_times, peak_values, kind='cubic', fill_value="extrapolate"
    # )
    # interpolated_time = np.linspace(abs_df['time'].min(), abs_df['time'].max(), 500)
    # interpolated_values = interpolation_func(interpolated_time)
    
    # # プロット
    # plt.figure(figsize=(12, 6))
    # plt.plot(abs_df['time'], abs_df['value'], label="Absolute Distance", color='blue')
    # plt.scatter(peak_times, peak_values, color='red', label="Local Maxima")
    # plt.plot(interpolated_time, interpolated_values, label="Interpolated Curve", color='green', linestyle='--')
    # plt.title("Local Maxima and Interpolated Curve")
    # plt.xlabel("Time")
    # plt.ylabel("Absolute Distance from Reference")
    # plt.legend()
    # plt.grid()
    # plt.show()


    # 極大値のインデックスを取得
    # peal_dict = dict(time = peak_indices, value = peak_values)
    # peaks_df = pd.DataFrame(peal_dict)
    # plot_signal_with_maxima_and_curve(signal_v, peaks_df)
    # return peaks_df
    

def plot_signal_with_maxima_and_curve(signal_v, peaks_df):
    """
    元の信号、極大値、および極大値を結んだ曲線をプロットする。

    Args:
        signal_v (pd.Series): 信号データ。
        peaks_df (pd.DataFrame): 極大値の時間（index）と値を含むデータフレーム。
    """
    # 時間軸
    time = signal_v.index

    # 極大値を補間
    spline_function = interp1d(peaks_df['time'], peaks_df['value'], kind='cubic', fill_value="extrapolate")
    smooth_curve = spline_function(time)

    # プロット
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal_v, label='Original Signal', color='blue', alpha=0.5)  # 元データ
    plt.scatter(peaks_df['time'], peaks_df['value'], color='red', label='Local Maxima', zorder=5)  # 極大値
    plt.plot(time, smooth_curve, label='Fitted Curve', color='orange')  # 補間曲線
    plt.title('Signal with Local Maxima and Fitted Curve')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

def find_change_point(signal_v, n_bkps):
    # コスト関数の設定
    model = "normal"
    # アルゴの設定と学習
    algo = rpt.Dynp(model=model).fit(signal_v)
    # 変化点の検出
    my_bkps = algo.predict(n_bkps=n_bkps)
    # plot
    rpt.display(signal_v, my_bkps, figsize=(12, 9))
    return plt

def make_graph(df, output_file_path):
    # グラフのプロット
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['x_acc'], label='x_acc', color='blue')
    plt.plot(df.index, df['y_acc'], label='y_acc', color='green')
    plt.plot(df.index, df['z_acc'], label='z_acc', color='red')

    # グラフのタイトルとラベル
    plt.title('Acceleration Data')
    plt.xlabel('Time (index)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid()

    save_graph_to_png(devided_plt, output_file_path)


def save_graph_to_png(plt, output_file_path):
    # グラフを保存
    plt.savefig(output_file_path)


for subfolder in subfolders:
    input_folder_path = os.path.join(base_input_folder, subfolder)
    output_folder_path = os.path.join(base_output_folder, subfolder)
    csv_files = glob.glob(os.path.join(input_folder_path, input_file_name))
    
    for file in csv_files:
        output_file_name = file.split('/')[-1].replace('.csv', '.png')
        output_file_path = os.path.join(output_folder_path, output_file_name)
        neko_df = pd.read_csv(file, header=None, names=['x_acc', 'y_acc', 'z_acc'])
        analyze_frequency_spectrum(neko_df)
