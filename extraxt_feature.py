import pandas as pd
import numpy as np
from scipy import signal
import glob
from collections import namedtuple
import os
import re

base_input_folder = 'data/bw_clip/'
base_output_folder = 'features'
subfolders = ['parkin', 'normal']
output_file_name =  'extracted_features.csv'
input_file_name = '*.csv'
all_results = []
group_nums = []
IDs = []
statistic = ['x_acc', 'y_acc', 'z_acc']

# def check_group(csv_files):
#     group_nums = []
#     for file in csv_files:
#         # 'normal' を含む場合は 1、それ以外は 0 を追加
#         if 'normal' in file:
#             group_nums.append(1)
#         else:
#             group_nums.append(0)
#     return group_nums

# def check_ID(csv_files):
#     transformed_names = []
#     for file in csv_files:
#         # 2種類のファイルパターンに基づいて名前を変換
#         if re.match(r"^\d+_neck\d+_clipout", file):
#             transformed_name = re.sub(r"(_neck)(\d)(_clipout)", r"-\2", file)
#         elif re.match(r"^[a-zA-Z]+\d+_neck\d+_clipout", file):
#             transformed_name = re.sub(r"(_neck)(\d)(_clipout)", r"-\2", file)
#         else:
#             # パターンに一致しない場合はファイル名をそのまま追加
#             transformed_name = file
#         transformed_names.append(transformed_name)
#     return transformed_names


def check_group(csv_files):
    for file in csv_files:
        if 'normal' in file:
            group_nums.append(1)
        else:
            group_nums.append(0)

def check_ID(csv_files):
    for file in csv_files:
        file_name = file.split('/')[-1].replace('.csv', '')
        if re.match(r"^\d{3}_neck\d", file_name):
            base, neck = file_name.split('_neck')
            IDs.append(f"{base}-{neck[0]}")
        elif re.match(r"^tone\d{3}_neck\d", file_name):
            base, neck = file_name.split('_neck')
            IDs.append(f"{base}-{neck[0]}")
        else:
            IDs.append(file_name)
    return IDs
        

def extract_features(csv_files):
    features_by_symptom = []

    for file in csv_files:
        df = pd.read_csv(file, header=None, names=statistic)
        two_features = extract_dict_features_from_csv(df)
        features_by_symptom.append(two_features)

    return features_by_symptom

def extract_dict_features_from_csv(df):
    basic_features = extract_basic_features(df[statistic])
    frequency_features = extract_frequency_features(df[statistic])
    two_features = {**basic_features, **frequency_features}
    return two_features

def extract_basic_features(df):
    features = {}

    for col in statistic:
        features[f"{col}__max"] = df[col].max()
        features[f"{col}__min"] = df[col].min()
        features[f"{col}__mean"] = df[col].mean()
        features[f"{col}__median"] = df[col].median()
        features[f"{col}__std"] = df[col].std()
        features[f"{col}__var"] = df[col].var()
        features[f"{col}__sum"] = df[col].sum()

    return features

def extract_frequency_features(df):
    # x, y, z軸のデータを取得
    x_data = df['x_acc'].to_numpy()
    y_data = df['y_acc'].to_numpy()
    z_data = df['z_acc'].to_numpy()

    # 特徴量を抽出
    features_x = calculate_frequency_features(x_data)
    features_y = calculate_frequency_features(y_data)
    features_z = calculate_frequency_features(z_data)

     # 各軸ごとに特徴量を辞書形式で作成
    combined_features = {}
    for axis, features in zip(statistic, [features_x, features_y, features_z]):
        for key, value in features.items():
            combined_features[f"{axis}__{key}"] = value

    return combined_features

def calculate_frequency_features(time_series_data, fs=100):
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

    # パーセンタイルの計算 (元の時系列データを使用)
    percentile25 = np.percentile(time_series_data, 25)
    percentile50 = np.percentile(time_series_data, 50)  # 中央値
    percentile75 = np.percentile(time_series_data, 75)

    return {
        'freezing_index': freezing_index,
        'central_frequency': weighted_frequency,
        'dominant_frequency': dominant_frequency,
        'amplitude': amplitude,
        'relative_amplitude': relative_amplitude,
        'percentile25': percentile25,
        'percentile50': percentile50,
        'percentile75': percentile75
    }

def save_features_to_csv(features, output_features_file):
    features.insert(0, 'group', group_nums)
    features.insert(0, 'ID', IDs)
    features.to_csv(output_features_file, index = False)



##################################################

for subfolder in subfolders:
    input_folder_path = os.path.join(base_input_folder, subfolder)
    output_features_file = os.path.join(base_output_folder, output_file_name)
    csv_files = glob.glob(os.path.join(input_folder_path, input_file_name))

    check_group(csv_files)
    check_ID(csv_files)
    features_by_symptom = extract_features(csv_files)
    all_results.append(pd.DataFrame(features_by_symptom))

final_features = pd.concat(all_results)
save_features_to_csv(final_features, output_features_file)
