import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import glob
import ruptures as rpt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy import interpolate
from pathlib import Path


base_input_folder = 'data/two_neck_dataset/'
fix_input_folder = 'data/clipout/'
base_output_folder = 'graph/change_point/'
csv_base_output_folder = 'data/clipout'
subfolders = ['normal', 'parkin']
ikikaeri = ['go', 'back']
input_file_name = '*.csv'
statistic = ['x_acc', 'y_acc', 'z_acc']
fix_info_file_path = 'fix_info.csv'
clip_info_file_path = 'clip_index.csv'
skip_point = 1000


change_point_buf = 0


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
    return max(positive_freqs[peaks])
    

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
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, signal_v, label='Original Signal', color='blue', alpha=0.5)  # 元データ
    # plt.scatter(peaks_df['time'], peaks_df['value'], color='red', label='Local Maxima', zorder=5)  # 極大値
    # plt.plot(time, smooth_curve, label='Fitted Curve', color='orange')  # 補間曲線
    # plt.title('Signal with Local Maxima and Fitted Curve')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    # plt.show()

def find_change_point(signal_v, n_bkps, isPoint):
    # コスト関数の設定
    model = "normal"
    algo = rpt.Dynp(model=model).fit(signal_v)
    my_bkps = algo.predict(n_bkps=n_bkps)
    # rpt.display(signal_v, my_bkps, figsize=(12, 9))
    if isPoint:
        return int((my_bkps[0] + my_bkps[1]) / 2)
    else:
        return my_bkps

def kyokudai_plot(df, output_folder_path, file_name):
    # データを原点に揃える
    df = df - df.iloc[0]

    # グラフのプロット
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['z_acc'], label='z_acc', color='green')

    # 極大値を検出
    max_points_x = []
    max_points_y = []
    for i in range(1, len(df) - 1):
        if df['z_acc'].iloc[i] > df['z_acc'].iloc[i - 1] and df['z_acc'].iloc[i] > df['z_acc'].iloc[i + 1]:
            max_points_x.append(df.index[i])
            max_points_y.append(df['z_acc'].iloc[i])

    # 極大値をプロット
    plt.scatter(max_points_x, max_points_y, color='C1', s=50, zorder=5)

    # スプライン補完を使って滑らかな曲線を描く
    if max_points_x:
        # 補完のためのスプライン関数を作成
        spline = interpolate.CubicSpline(max_points_x, max_points_y)

        # 補完曲線を描画するための新しいx値を生成
        new_x = np.linspace(min(max_points_x), max(max_points_x), 500)
        new_y = spline(new_x)

        # 補完曲線を描画
        plt.plot(new_x, new_y, color='orange', label='Spline Interpolation', linewidth=2)

    # グラフの仕上げ
    plt.xlabel('Time')
    plt.ylabel('z_acc')
    plt.legend()
    plt.grid(True)
    plt.title('Curve with Spline Interpolation')
    plt.show()

  

def plot_curve(df, output_folder_path, file_name,  peakHeltz):
    df = df - df.iloc[0]

    period = 1 / peakHeltz * 100   # 2Hzの周期（0.5秒間隔）
    times = np.arange(0, df.index[-1], period)  # df.indexの最終値までの0.5秒刻み

    # グラフのプロット
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['z_acc'], label='z_acc', color='green')

    max_points_x = []
    max_points_y = []


    # 最大値を取る点をプロット
    for start_time in times:
        end_time = start_time + period
        # この期間のデータを取得
        mask = (df.index >= start_time) & (df.index < end_time)
        period_data = df[mask]
        
        # z_accの最大値とそのインデックスを取得
        if not period_data.empty:
            max_index = period_data['z_acc'].idxmax()  # 最大値のインデックス
            max_value = period_data['z_acc'].max()  # 最大値
            
            # 最大値をプロット
            plt.scatter(max_index, max_value, color='C1', s=50, zorder=5)
            
            # 最大値の座標をリストに追加
            max_points_x.append(max_index)
            max_points_y.append(max_value)
    
    # スプライン補完を使って滑らかな曲線を描く
    if max_points_x:
        # 補完のためのスプライン関数を作成
        spline = interpolate.CubicSpline(max_points_x, max_points_y)

        # 補完曲線を描画するための新しいx値を生成
        new_x = np.linspace(min(max_points_x), max(max_points_x), 500)
        new_y = spline(new_x)

        # 補完曲線を描画
        plt.plot(new_x, new_y, color='orange', label='Spline Interpolation', linewidth=2)

    # # グラフのタイトルとラベル
    # plt.plot(df.index, df['z_acc'], label='z_acc', color='green')
    # plt.title('Acceleration Data with 2Hz Vertical Lines')
    # plt.xlabel('Time (index)')
    # plt.ylabel('Acceleration')
    # plt.legend()
    # plt.grid(True)

    # # グラフを表示
    # plt.show()

    # save_graph_to_png(plt, output_folder_path, file_name, )

def make_graph(df, change_point, start_end, output_folder_path, file_name):
    df = df - df.iloc[0]
    start_point = start_end[0]
    end_point = start_end[1]
    save_graph_to_png(df[start_point:end_point], output_folder_path, file_name)


def devide_graph(df, change_point, start_end, csv_output_folder_path, output_folder_path, file_name):
    start_point = start_end[0]
    end_point = start_end[1]
    save_csv_slice(df, start_point, change_point + skip_point, csv_output_folder_path, output_folder_path, file_name, 'go')
    save_csv_slice(df, change_point + skip_point + 1, end_point, csv_output_folder_path, output_folder_path, file_name, 'back')


def show_graph(df, title):
    plt.plot(df.index, df['x_acc'], label='x_acc', color="red")
    plt.plot(df.index, df['y_acc'], label='y_acc', color="green")
    plt.plot(df.index, df['z_acc'], label='z_acc', color="blue")
    plt.title(title)
    # plt.xlabel('Time (index)')
    # plt.ylabel('Acceleration')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    plt.savefig('graph/presentation/' + title + '.png')
    plt.close()


def save_graph_to_png(df,  output_folder_path, file_name):
    plt.plot(df.index, df['x_acc'], label='x_acc', color="red")
    plt.plot(df.index, df['y_acc'], label='y_acc', color="green")
    plt.plot(df.index, df['z_acc'], label='z_acc', color="blue")
    plt.xlabel('Time (index)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    output_folder_path = os.path.join(output_folder_path, file_name + '.png')
    plt.savefig(output_folder_path)


def save_csv_slice(df, start, end, csv_folder_path, output_folder_path, file_name, file_suffix):
    ikikaeri_name = file_name + '_' + file_suffix
    df_slice = df.loc[start:end, statistic].reset_index(drop=True)
    # save_graph_to_png(df_slice, output_folder_path, ikikaeri_name)
    csv_output_file_path = os.path.join(csv_folder_path, ikikaeri_name + '.csv')
    df_slice.to_csv(csv_output_file_path, index = False) 

def fix_csv_slice(df, start, end, symptom, file_name):
    df_slice = df.loc[start:end, statistic].reset_index(drop=True)
    csv_output_file_path = os.path.join(f"data/fix/{symptom}", file_name + ".csv")
    df_slice.to_csv(csv_output_file_path, index = False) 
    print(f'{csv_output_file_path}で保存したぜ')
    # show_graph(df_slice, file_name)

def check_change_point(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name + "_go" + ".csv")
    if os.path.exists(file_path):
        yomitori_df = pd.read_csv(file_path, names=statistic, skiprows=1)

    return len(yomitori_df)



def merge_csv(folder_path, file_name):
    # 001_neck1_clipoutをもらった。
    # 001_neck1_clipout_go, backの2つを取得して保存。
    # 結合してreturn 
    csv_files = []
    for move in ikikaeri:
        file_path = os.path.join(folder_path, file_name + "_" + move + ".csv")
        if os.path.exists(file_path):
            yomitori_df = pd.read_csv(file_path, names=statistic, skiprows=1)
            csv_files.append(yomitori_df)
        else:
            raise FileNotFoundError(f"File {file_path} not found.")
    
    merged_df = pd.concat(csv_files).reset_index(drop=True)
    return merged_df

    # plt.figure(figsize=(12, 6))
    # plt.title(file_name)
    # plt.plot(merged_df.index, merged_df['z_acc'], label='z_acc', color='green')
    # plt.plot(merged_df.index, merged_df['y_acc'], label='y_acc', color='red')
    # plt.plot(merged_df.index, merged_df['x_acc'], label='x_acc', color='blue')
    # plt.axvline(center, color="orange", lw=2, linestyle='--', label="center")
    # plt.legend()
    # plt.xticks(np.arange(0, 4000, step=200))
    # plt.grid()
    # plt.savefig('graph/merged_csv/' + file_name + ".png")
    # plt.close()

def fixxx_graph(symptom):
    # 空白ぎょうなら読み飛ばす
    info_df = pd.read_csv(fix_info_file_path, names=['name', 'group', 'center', 'end'], skiprows=1).dropna(how='all')
    if symptom == 'parkin':
        info_df = info_df[info_df['group'] == 1]
    else:
        info_df = info_df[info_df['group'] == 0]

    for _, gyo in info_df.iterrows():
        file_name = os.path.splitext(gyo['name'])[0]
        folder_path = os.path.join(fix_input_folder, symptom)    

        merged_df = merge_csv(folder_path, file_name)
        center = check_change_point(folder_path, file_name)
        end = len(merged_df)

        if pd.notnull(gyo['center']):  # center が存在する場合
            center = int(gyo['center'])

        if pd.notnull(gyo['end']):  # end が存在する場合
            end = int(gyo['end'])
        
        # スライスして保存
        for idx, move in enumerate(ikikaeri):
            start = 0 if idx == 0 else center + 1
            stop = center if idx == 0 else end
            fix_csv_slice(merged_df, start, stop, symptom, file_name + '_' + move)
    




# 修正情報を適用するメイン関数
# def fix_graph(symptom):
#     # 修正情報ファイルを読み込み
#     info_df = pd.read_csv(fix_info_file_path, names=['id', 'group', 'center', 'end']).dropna(how='all')

#     if symptom == 'parkin':
#         info_df = info_df[info_df['group'] == 1]
#         file_format = "tone{0}_neck{1}_clipout_{2}.csv"    
#     else:
#         info_df = info_df[info_df['group'] == 0]
#         file_format = "{0}_neck{1}_clipout_{2}.csv"  
    
#     for _, gyo in info_df.iterrows():
#         file_id = gyo['id']
#         splited_id, splited_num = file_id.split('-')
#         folder_path = os.path.join(fix_input_folder, symptom)    

#         csv_files = []
#         for move in ikikaeri:
#             file_name = file_format.format(splited_id, splited_num, move)
#             file_path = os.path.join(folder_path, file_name)
#             if os.path.exists(file_path):
#                 csv_files.append(pd.read_csv(file_path, names=statistic, skiprows=1))
#             else:
#                 raise FileNotFoundError(f"File {file_path} not found.")
            
#         marged_csv = pd.concat(csv_files).reset_index(drop=True)
#         show_graph(marged_csv, file_name)
        # center = len(csv_files[0])
        # end = len(marged_csv)
        # if pd.notnull(gyo['center']):  # center が存在する場合
        #     center = int(gyo['center'])

        # if pd.notnull(gyo['end']):  # end が存在する場合
        #     end = int(gyo['end'])
        
        # # スライスして保存
        # for idx, move in enumerate(ikikaeri):
        #     start = 0 if idx == 0 else center + 1
        #     stop = center if idx == 0 else end
        #     fix_csv_slice(marged_csv, start, stop, symptom, file_name)
    
        # グラフをプロット
        # plt.figure()
        # plt.title(file_id)
        # plt.plot(marged_csv.index[:end], marged_csv['z_acc'][:end], label='z_acc', color='green')
        # plt.plot(marged_csv.index[:end], marged_csv['y_acc'][:end], label='y_acc', color='red')
        # plt.plot(marged_csv.index[:end], marged_csv['x_acc'][:end], label='x_acc', color='blue')
        # plt.axvline(center, color="orange", lw=2, linestyle='--', label="center")
        # plt.legend()
        # plt.show()


def clipout_data(df, symptom):
    group_num = 0 if symptom == 'normal' else 1
    clip_info_df = pd.read_csv(clip_info_file_path, names=['name', 'start',  'end', 'group'])
    clip_info_df = clip_info_df[clip_info_df['group'] == group_num]
    
    for _, gyo in clip_info_df.iterrows():
        file_name = gyo['name']
        start = gyo['start']
        end = gyo['end']
        
        folder_path = os.path.join(base_input_folder, symptom)   
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path, names=statistic)
        else:
            break 

        df[start:end].to_csv(f'data/bw_clip/{symptom}/{file_name}', index=False, header=False,)



for subfolder in subfolders:
    input_folder_path = os.path.join(base_input_folder, subfolder)
    output_folder_path = os.path.join(base_output_folder, subfolder)
    csv_output_folder_path = os.path.join(csv_base_output_folder, subfolder)
    csv_files = glob.glob(os.path.join(input_folder_path, input_file_name))
    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file, header=None, names=['x_acc', 'y_acc', 'z_acc'])
        show_graph(df, file_name)




    # fixxx_graph(subfolder)
    
    # for file in csv_files:
    #     file_name = os.path.splitext(os.path.basename(file))[0]
    # #     filtered_df = pd.read_csv(file, header=None, names=['x_acc', 'y_acc', 'z_acc'], skiprows=skip_point, skipfooter=1000)
    #     df = pd.read_csv(file, header=None, names=['x_acc', 'y_acc', 'z_acc'])
    #     gyo = len(df)
    #     if(gyo < 700):
    #         print(f'{file_name} ... {gyo}行')


    #     change_point = find_change_point(filtered_df, 2, True)
    #     start_end = find_change_point(df, 2, False)
    #     # make_graph(df, change_point, start_end, output_folder_path, file_name)
    #     devide_graph(df, change_point, start_end, csv_output_folder_path, output_folder_path, file_name)


    # for file in csv_files:
    #     df = pd.read_csv(file, header=None, names=['x_acc', 'y_acc', 'z_acc'], skiprows=1)
    #     file_name = Path(file).stem
    #     file_name = file_name.replace("_go", "")
    #     file_name = file_name.replace("_back", "")

    #     merge_csv(f'data/clipout/{subfolder}/', file_name)



# ファイル取得→goとかbackのける。


