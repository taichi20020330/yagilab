# 各被験者のセンサ×統計量のファイル（○○_feature_cut.csv）を個別に作る

import pandas as pd
import numpy as np
import datetime
import glob
import csv
import matplotlib.pyplot as plt
from scipy import signal

#default
#file_list = glob.glob("..\\data\\elderly\\*.csv")

#edited
file_list = glob.glob("csv/Parkinson/2023/eskin/*.csv")

file_list = sorted(file_list)

variable_list = ['LeftFootIMU_acc_x', 'LeftFootIMU_acc_y', 'LeftFootIMU_acc_z',
                 'LeftFootIMU_gyr_x', 'LeftFootIMU_gyr_y', 'LeftFootIMU_gyr_z',
                 'LeftLowerLegIMU_acc_x', 'LeftLowerLegIMU_acc_y', 'LeftLowerLegIMU_acc_z',
                 'LeftLowerLegIMU_gyr_x', 'LeftLowerLegIMU_gyr_y', 'LeftLowerLegIMU_gyr_z',
                 'LeftUpperLegIMU_acc_x', 'LeftUpperLegIMU_acc_y', 'LeftUpperLegIMU_acc_z',
                 'LeftUpperLegIMU_gyr_x', 'LeftUpperLegIMU_gyr_y', 'LeftUpperLegIMU_gyr_z',
                 'PelvisIMU_acc_x', 'PelvisIMU_acc_y', 'PelvisIMU_acc_z',
                 'PelvisIMU_gyr_x', 'PelvisIMU_gyr_y', 'PelvisIMU_gyr_z',
                 'RightFootIMU_acc_x', 'RightFootIMU_acc_y', 'RightFootIMU_acc_z',
                 'RightFootIMU_gyr_x', 'RightFootIMU_gyr_y', 'RightFootIMU_gyr_y',
                 'RightLowerLegIMU_acc_x', 'RightLowerLegIMU_acc_y', 'RightLowerLegIMU_acc_z',
                 'RightLowerLegIMU_gyr_x', 'RightLowerLegIMU_gyr_y', 'RightLowerLegIMU_gyr_z',
                 'RightUpperLegIMU_acc_x', 'RightUpperLegIMU_acc_y', 'RightUpperLegIMU_acc_z',
                 'RightUpperLegIMU_gyr_x', 'RightUpperLegIMU_gyr_y', 'RightUpperLegIMU_gyr_z'
                 ]


#default
#df_periods = pd.read_excel("..\\data\\wperiods.xlsx", sheet_name = 'elderly')
#df_height_list = pd.read_excel("csv\\height_list.xlsx", skiprows = 0, sheet_name = 'elderly')

#edited

#elderly
#df_periods = pd.read_excel("csv/Parkinson/2023/index.xlsx", sheet_name = 'elderly')
#df_height_list = pd.read_excel("csv/height_list_2023.xlsx", skiprows = 0, sheet_name = 'elderly')

#parkins
df_periods = pd.read_excel("csv/Parkinson/2023/index.xlsx", sheet_name = 'parkinson')
df_height_list = pd.read_excel("csv/height_list_2023.xlsx", skiprows = 0, sheet_name = 'parkinson')

for path, count in zip(file_list, range(len(file_list))):
    ex_id = path.split('\\')[-1].split('.')[0]

    #default
    #file_name = "csv\\elderly_cut_flt\\" + ex_id + "_feature_cut_flt.csv"

    #edited
    ex_id = ex_id[25:]

    if 'a' in ex_id:
        ex_id = ex_id.replace('a', '-1')
    if 'b' in ex_id:
        ex_id = ex_id.replace('b', '-2')

    file_name = "csv/Parkinson_cut_flt/2023/" + ex_id + "_feature_cut_flt.csv"
    
    #elderlyの読み込み
    #df = pd.read_csv(path, skiprows = 7)
    
    #Parkinsonの読み込み
    if count > 53:
        df = pd.read_csv(path, skiprows = 7)
        x_name = 'Timestamp'
    else:
        df = pd.read_csv(path, skiprows = 0)
        x_name = 'Time'

    # 時間の切り出し
    walk_start = df_periods['walk start'][count]
    turn_start = df_periods['turn start'][count]
    turn_end = df_periods['turn end'][count]
    walk_end = df_periods['walk end'][count]

    # あるきはじめから旋回開始
    df1 = df.loc[walk_start:turn_start]
    # 旋回終わりから
    df2 = df.loc[turn_end:walk_end]
    df = pd.concat([df.loc[walk_start:turn_start], df.loc[turn_end:walk_end]])
    df = df.reset_index()
    print([walk_start, turn_start, turn_end, walk_end])
    
    power_LF = []
    power_LLL = []
    power_LUL = []
    power_Pelvis = []
    power_RF = []
    power_RLL = []
    power_RUL = []
    
    #median filter　ノイズを消す　直近のデータ5こみる
    LF_acc_x_flt = signal.medfilt(df['LeftFootIMU_acc_x'], kernel_size = 5)
    LF_acc_y_flt = signal.medfilt(df['LeftFootIMU_acc_y'], kernel_size = 5)
    LF_acc_z_flt = signal.medfilt(df['LeftFootIMU_acc_z'], kernel_size = 5)
    LF_gyr_x_flt = signal.medfilt(df['LeftFootIMU_gyr_x'], kernel_size = 5)
    LF_gyr_y_flt = signal.medfilt(df['LeftFootIMU_gyr_y'], kernel_size = 5)
    LF_gyr_z_flt = signal.medfilt(df['LeftFootIMU_gyr_z'], kernel_size = 5)
    LLL_acc_x_flt = signal.medfilt(df['LeftLowerLegIMU_acc_x'], kernel_size = 5)
    LLL_acc_y_flt = signal.medfilt(df['LeftLowerLegIMU_acc_y'], kernel_size = 5)
    LLL_acc_z_flt = signal.medfilt(df['LeftLowerLegIMU_acc_z'], kernel_size = 5)
    LLL_gyr_x_flt = signal.medfilt(df['LeftLowerLegIMU_gyr_x'], kernel_size = 5)
    LLL_gyr_y_flt = signal.medfilt(df['LeftLowerLegIMU_gyr_y'], kernel_size = 5)
    LLL_gyr_z_flt = signal.medfilt(df['LeftLowerLegIMU_gyr_z'], kernel_size = 5)
    LUL_acc_x_flt = signal.medfilt(df['LeftUpperLegIMU_acc_x'], kernel_size = 5)
    LUL_acc_y_flt = signal.medfilt(df['LeftUpperLegIMU_acc_y'], kernel_size = 5)
    LUL_acc_z_flt = signal.medfilt(df['LeftUpperLegIMU_acc_z'], kernel_size = 5)
    LUL_gyr_x_flt = signal.medfilt(df['LeftUpperLegIMU_gyr_x'], kernel_size = 5)
    LUL_gyr_y_flt = signal.medfilt(df['LeftUpperLegIMU_gyr_y'], kernel_size = 5)
    LUL_gyr_z_flt = signal.medfilt(df['LeftUpperLegIMU_gyr_z'], kernel_size = 5)
    Pelvis_acc_x_flt = signal.medfilt(df['PelvisIMU_acc_x'], kernel_size = 5)
    Pelvis_acc_y_flt = signal.medfilt(df['PelvisIMU_acc_y'], kernel_size = 5)
    Pelvis_acc_z_flt = signal.medfilt(df['PelvisIMU_acc_z'], kernel_size = 5)
    Pelvis_gyr_x_flt = signal.medfilt(df['PelvisIMU_gyr_x'], kernel_size = 5)
    Pelvis_gyr_y_flt = signal.medfilt(df['PelvisIMU_gyr_y'], kernel_size = 5)
    Pelvis_gyr_z_flt = signal.medfilt(df['PelvisIMU_gyr_z'], kernel_size = 5)
    RF_acc_x_flt = signal.medfilt(df['RightFootIMU_acc_x'], kernel_size = 5)
    RF_acc_y_flt = signal.medfilt(df['RightFootIMU_acc_y'], kernel_size = 5)
    RF_acc_z_flt = signal.medfilt(df['RightFootIMU_acc_z'], kernel_size = 5)
    RF_gyr_x_flt = signal.medfilt(df['RightFootIMU_gyr_x'], kernel_size = 5)
    RF_gyr_y_flt = signal.medfilt(df['RightFootIMU_gyr_y'], kernel_size = 5)
    RF_gyr_z_flt = signal.medfilt(df['RightFootIMU_gyr_z'], kernel_size = 5)
    RLL_acc_x_flt = signal.medfilt(df['RightLowerLegIMU_acc_x'], kernel_size = 5)
    RLL_acc_y_flt = signal.medfilt(df['RightLowerLegIMU_acc_y'], kernel_size = 5)
    RLL_acc_z_flt = signal.medfilt(df['RightLowerLegIMU_acc_z'], kernel_size = 5)
    RLL_gyr_x_flt = signal.medfilt(df['RightLowerLegIMU_gyr_x'], kernel_size = 5)
    RLL_gyr_y_flt = signal.medfilt(df['RightLowerLegIMU_gyr_y'], kernel_size = 5)
    RLL_gyr_z_flt = signal.medfilt(df['RightLowerLegIMU_gyr_z'], kernel_size = 5)
    RUL_acc_x_flt = signal.medfilt(df['RightUpperLegIMU_acc_x'], kernel_size = 5)
    RUL_acc_y_flt = signal.medfilt(df['RightUpperLegIMU_acc_y'], kernel_size = 5)
    RUL_acc_z_flt = signal.medfilt(df['RightUpperLegIMU_acc_z'], kernel_size = 5)
    RUL_gyr_x_flt = signal.medfilt(df['RightUpperLegIMU_gyr_x'], kernel_size = 5)
    RUL_gyr_y_flt = signal.medfilt(df['RightUpperLegIMU_gyr_y'], kernel_size = 5)
    RUL_gyr_z_flt = signal.medfilt(df['RightUpperLegIMU_gyr_z'], kernel_size = 5)

    # 一旦平均求めとく
    mean_LF_acc_x_flt = np.mean(LF_acc_x_flt)
    mean_LF_acc_y_flt = np.mean(LF_acc_y_flt)
    mean_LF_acc_z_flt = np.mean(LF_acc_z_flt)
    mean_LLL_acc_x_flt = np.mean(LLL_acc_x_flt)
    mean_LLL_acc_y_flt = np.mean(LLL_acc_y_flt)
    mean_LLL_acc_z_flt = np.mean(LLL_acc_z_flt)
    mean_LUL_acc_x_flt = np.mean(LUL_acc_x_flt)
    mean_LUL_acc_y_flt = np.mean(LUL_acc_y_flt)
    mean_LUL_acc_z_flt = np.mean(LUL_acc_z_flt)
    mean_Pelvis_acc_x_flt = np.mean(Pelvis_acc_x_flt)
    mean_Pelvis_acc_y_flt = np.mean(Pelvis_acc_y_flt)
    mean_Pelvis_acc_z_flt = np.mean(Pelvis_acc_z_flt)
    mean_RF_acc_x_flt = np.mean(RF_acc_x_flt)
    mean_RF_acc_y_flt = np.mean(RF_acc_y_flt)
    mean_RF_acc_z_flt = np.mean(RF_acc_z_flt)
    mean_RLL_acc_x_flt = np.mean(RLL_acc_x_flt)
    mean_RLL_acc_y_flt = np.mean(RLL_acc_y_flt)
    mean_RLL_acc_z_flt = np.mean(RLL_acc_z_flt)
    mean_RUL_acc_x_flt = np.mean(RUL_acc_x_flt)
    mean_RUL_acc_y_flt = np.mean(RUL_acc_y_flt)
    mean_RUL_acc_z_flt = np.mean(RUL_acc_z_flt)
    
    flt_name_list = ['LeftFootIMU_acc_x', 'LeftFootIMU_acc_y', 'LeftFootIMU_acc_z',
                     'LeftFootIMU_gyr_x', 'LeftFootIMU_gyr_y', 'LeftFootIMU_gyr_z',
                     'LeftLowerLegIMU_acc_x', 'LeftLowerLegIMU_acc_y', 'LeftLowerLegIMU_acc_z',
                     'LeftLowerLegIMU_gyr_x', 'LeftLowerLegIMU_gyr_y', 'LeftLowerLegIMU_gyr_z',
                     'LeftUpperLegIMU_acc_x', 'LeftUpperLegIMU_acc_y', 'LeftUpperLegIMU_acc_z',
                     'LeftUpperLegIMU_gyr_x', 'LeftUpperLegIMU_gyr_y', 'LeftUpperLegIMU_gyr_z',
                     'PelvisIMU_acc_x', 'PelvisIMU_acc_y', 'PelvisIMU_acc_z',
                     'PelvisIMU_gyr_x', 'PelvisIMU_gyr_y', 'PelvisIMU_gyr_z',
                     'RightFootIMU_acc_x', 'RightFootIMU_acc_y', 'RightFootIMU_acc_z',
                     'RightFootIMU_gyr_x', 'RightFootIMU_gyr_y', 'RightFootIMU_gyr_y',
                     'RightLowerLegIMU_acc_x', 'RightLowerLegIMU_acc_y', 'RightLowerLegIMU_acc_z',
                     'RightLowerLegIMU_gyr_x', 'RightLowerLegIMU_gyr_y', 'RightLowerLegIMU_gyr_z',
                     'RightUpperLegIMU_acc_x', 'RightUpperLegIMU_acc_y', 'RightUpperLegIMU_acc_z',
                     'RightUpperLegIMU_gyr_x', 'RightUpperLegIMU_gyr_y', 'RightUpperLegIMU_gyr_z']
    flt_variable_list = [LF_acc_x_flt, LF_acc_y_flt, LF_acc_z_flt,
                         LF_gyr_x_flt, LF_gyr_y_flt, LF_gyr_z_flt,
                         LLL_acc_x_flt, LLL_acc_y_flt, LLL_acc_z_flt,
                         LLL_gyr_x_flt, LLL_gyr_y_flt, LLL_gyr_z_flt,
                         LUL_acc_x_flt, LUL_acc_y_flt, LUL_acc_z_flt,
                         LUL_gyr_x_flt, LUL_gyr_y_flt, LUL_gyr_z_flt,
                         Pelvis_acc_x_flt, Pelvis_acc_y_flt, Pelvis_acc_z_flt,
                         Pelvis_gyr_x_flt, Pelvis_gyr_y_flt, Pelvis_gyr_z_flt,
                         RF_acc_x_flt, RF_acc_y_flt, RF_acc_z_flt,
                         RF_gyr_x_flt, RF_gyr_y_flt, RF_gyr_z_flt,
                         RLL_acc_x_flt, RLL_acc_y_flt, RLL_acc_z_flt,
                         RLL_gyr_x_flt, RLL_gyr_y_flt, RLL_gyr_z_flt,
                         RUL_acc_x_flt, RUL_acc_y_flt, RUL_acc_z_flt,
                         RUL_gyr_x_flt, RUL_gyr_y_flt, RUL_gyr_z_flt]
    


    for n in range(len(LF_acc_x_flt)):
        power_LF.append(np.square(LF_acc_x_flt[n] - mean_LF_acc_x_flt) + 
                        np.square(LF_acc_y_flt[n] - mean_LF_acc_y_flt) +
                        np.square(LF_acc_z_flt[n] - mean_LF_acc_z_flt))
        power_LLL.append(np.square(LLL_acc_x_flt[n] - mean_LLL_acc_x_flt) + 
                         np.square(LLL_acc_y_flt[n] - mean_LLL_acc_y_flt) +
                         np.square(LLL_acc_z_flt[n] - mean_LLL_acc_z_flt))
        power_LUL.append(np.square(LUL_acc_x_flt[n] - mean_LUL_acc_x_flt) + 
                         np.square(LUL_acc_y_flt[n] - mean_LUL_acc_y_flt) +
                         np.square(LUL_acc_z_flt[n] - mean_LUL_acc_z_flt))
        power_Pelvis.append(np.square(Pelvis_acc_x_flt[n] - mean_Pelvis_acc_x_flt) + 
                            np.square(Pelvis_acc_y_flt[n] - mean_Pelvis_acc_y_flt) +
                            np.square(Pelvis_acc_z_flt[n] - mean_Pelvis_acc_z_flt))
        power_RF.append(np.square(RF_acc_x_flt[n] - mean_RF_acc_x_flt) + 
                        np.square(RF_acc_y_flt[n] - mean_RF_acc_y_flt) +
                        np.square(RF_acc_z_flt[n] - mean_RF_acc_z_flt))
        power_RLL.append(np.square(RLL_acc_x_flt[n] - mean_RLL_acc_x_flt) + 
                         np.square(RLL_acc_y_flt[n] - mean_RLL_acc_y_flt) +
                         np.square(RLL_acc_z_flt[n] - mean_RLL_acc_z_flt))
        power_RUL.append(np.square(RUL_acc_x_flt[n] - mean_RUL_acc_x_flt) + 
                         np.square(RUL_acc_y_flt[n] - mean_RUL_acc_y_flt) +
                         np.square(RUL_acc_z_flt[n] - mean_RUL_acc_z_flt))
    power_name_list = ['LeftFoot_mag', 'LeftLowerLeg_mag', 'LeftUpperLeg_mag', 'Pelvis_mag',
                       'RightFoot_mag', 'RightLowerLeg_mag', 'RightUpperLeg_mag']
    power_variable_list = [power_LF, power_LLL, power_LUL, power_Pelvis,
                  power_RF, power_RLL, power_RUL]
    
    # mark_point = [walk_start, turn_start, turn_end, walk_end]
    # print(mark_point)
    # file_name2 = "image\\Parkinson_auto\\" + ex_id + "_auto_detected.png"
    # fig = plt.figure()
    # h1, = plt.plot(df[x_name], df['LeftUpperLegIMU_acc_x'], label = 'LUL_acc_x', marker="*", markersize=10, markevery = mark_point, markerfacecolor="r")
    # plt.xlabel("Time[s]")
    # plt.ylabel("Acceleration[m/s^2]")
    # plt.legend(handles = [h1])
    # plt.ylim(-25, 25)
    # fig.savefig(file_name2)
    
    # 関節角度
    angle_name_list = ['LeftAnkle_x', 'LeftAnkle_y', 'LeftAnkle_z',
                       'LeftKnee_x', 'LeftKnee_y', 'LeftKnee_z',
                       'LeftHip_x', 'LeftHip_y', 'LeftHip_z',
                       'RightHip_x', 'RightHip_y', 'RightHip_z',
                       'RightAnkle_x', 'RightAnkle_y', 'RightAnkle_z',
                       'RightKnee_x', 'RightKnee_y', 'RightKnee_z']
    
    angle_variable_list = []
    
    for name in angle_name_list:
        angle_variable_list.append(df[name])

    # 歩幅の計算
    step_Ankle = []
    step_Knee = []
    step_Hip = []
    # 身長(m)
    height = df_height_list[df_height_list['ID'] == ex_id]['Height'].values[0] / 100
    for n in range(len(LF_acc_x_flt)):
        step_Ankle.append(np.sqrt(np.square(df.loc[n, 'LeftAnkle_g_x'] - df.loc[n, 'RightAnkle_g_x']) +
                          np.square(df.loc[n, 'LeftAnkle_g_y'] - df.loc[n, 'RightAnkle_g_y']) +
                          np.square(df.loc[n, 'LeftAnkle_g_z'] - df.loc[n, 'RightAnkle_g_z'])) / height)
        step_Knee.append(np.sqrt(np.square(df.loc[n, 'LeftKnee_g_x'] - df.loc[n, 'RightKnee_g_x']) +
                          np.square(df.loc[n, 'LeftKnee_g_y'] - df.loc[n, 'RightKnee_g_y']) +
                          np.square(df.loc[n, 'LeftKnee_g_z'] - df.loc[n, 'RightKnee_g_z'])) / height)
        step_Hip.append(np.sqrt(np.square(df.loc[n, 'LeftHip_g_x'] - df.loc[n, 'RightHip_g_x']) +
                          np.square(df.loc[n, 'LeftHip_g_y'] - df.loc[n, 'RightHip_g_y']) +
                          np.square(df.loc[n, 'LeftHip_g_z'] - df.loc[n, 'RightHip_g_z'])) / height)
    step_name_list = ['Ankle_step', 'Knee_step', 'Hip_step']
    step_variable_list = [step_Ankle, step_Knee, step_Hip]
    


    # ↑でーたをきれいにする　ここから特徴量の処理

    
    with open(file_name, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['variable name', 'max', 'min', 'mean', 'var', '25th percentile', '50th percentile', '75th percentile', 
                         'freezing index', 'central frequency', 'dominant frequency', 'amplitude', 'relative amplitude'])
        for name, variable in zip(flt_name_list + power_name_list + angle_name_list + step_name_list, flt_variable_list + power_variable_list + angle_variable_list + step_variable_list):
            #時間領域の特徴量　基本的なやつ
            max = np.max(variable)
            min = np.min(variable)
            mean = np.mean(variable)
            var = np.var(variable)
            percentile25 = np.percentile(variable, 25)
            percentile50 = np.percentile(variable, 50)
            percentile75 = np.percentile(variable, 75)
            
            #周波数領域の特徴量　スペクトル　周波数　行きの部分、帰りの部分の平均
            fs = 100
            y1 = variable[:len(df1['LeftFootIMU_acc_x'])]
            y1_detrended = signal.detrend(y1)
            y2 = variable[len(df1['LeftFootIMU_acc_x']):]
            y2_detrended = signal.detrend(y2)
            
            freq1, P1 = signal.welch(y1_detrended, fs, nperseg = 512, detrend = False)
            freq2, P2 = signal.welch(y2_detrended, fs, nperseg = 512, detrend = False)

            P_mean = np.mean([P1,P2], axis = 0)
            
            #FIの計算　　パーキンソン病のひとだとFIの値が有名らしい

            locomotor_band_power = []
            freezing_band_power = []
            for freq, P in zip(freq1, P_mean):
                if(0.5 <= freq < 3):
                    locomotor_band_power.append(P)
                elif(3 <= freq < 8):
                    freezing_band_power.append(P)
            freezing_index = np.sum(freezing_band_power) / np.sum(locomotor_band_power)
            
            #CFの計算
            weighted_frequency = []
            for freq, P in zip(freq1, P_mean):
                weighted_frequency.append(freq * (P / np.sum(P_mean)))
            central_frequency = np.sum(weighted_frequency)

            #DFの計算
            peaks, _ = signal.find_peaks(P_mean)
            peak_max = np.argmax(P_mean[peaks])
            dominant_frequency = freq1[peaks[peak_max]]
            
            #振幅の計算
            amplitude = np.max(P_mean)
            
            #相対的な振幅の計算
            relative_amplitude = np.max(P_mean) / np.sum(P_mean)

            # csvに書き込む
            writer.writerow([name, max, min, mean, var, percentile25, percentile50, percentile75, freezing_index, central_frequency, dominant_frequency, amplitude, relative_amplitude])
            
        # for name, variable in zip(power_name_list, power_variable_list):
        #     #時間領域の特徴量
        #     max = np.max(variable)
        #     min = np.min(variable)
        #     mean = np.mean(variable)
        #     var = np.var(variable)
        #     percentile25 = np.percentile(variable, 25)
        #     percentile50 = np.percentile(variable, 50)
        #     percentile75 = np.percentile(variable, 75)
            
        #     #周波数領域の特徴量
        #     fs = 100
        #     y1 = variable[:len(df1['LeftFootIMU_acc_x'])]
        #     y1_detrended = signal.detrend(y1)
        #     y2 = variable[len(df1['LeftFootIMU_acc_x']):]
        #     y2_detrended = signal.detrend(y2)
            
        #     freq1, P1 = signal.welch(y1_detrended, fs, nperseg = 512, detrend = False)
        #     freq2, P2 = signal.welch(y2_detrended, fs, nperseg = 512, detrend = False)

        #     P_mean = np.mean([P1, P2], axis=0)
            
        #     #FIの計算
        #     locomotor_band_power = []
        #     freezing_band_power = []
        #     for freq, P in zip(freq1, P_mean):
        #         if(0.5 <= freq < 3):
        #             locomotor_band_power.append(P)
        #         elif(3 <= freq < 8):
        #             freezing_band_power.append(P)
        #     freezing_index = np.sum(freezing_band_power) / np.sum(locomotor_band_power)
            
        #     #CFの計算
        #     weighted_frequency = []
        #     for freq, P in zip(freq1, P_mean):
        #         weighted_frequency.append(freq * (P / np.sum(P_mean)))
        #     central_frequency = np.sum(weighted_frequency)

        #     #DFの計算
        #     peaks, _ = signal.find_peaks(P_mean)
        #     peak_max = np.argmax(P_mean[peaks])
        #     dominant_frequency = freq1[peaks[peak_max]]
            
        #     #振幅の計算
        #     amplitude = np.max(P_mean)
            
        #     #相対的な振幅の計算
        #     relative_amplitude = np.max(P_mean) / np.sum(P_mean)
            
        #     writer.writerow([name, max, min, mean, var, percentile25, percentile50, percentile75, freezing_index, central_frequency, dominant_frequency, amplitude, relative_amplitude])
    
