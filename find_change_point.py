import numpy as np
import ruptures as rpt
from ruptures.metrics import precision_recall
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 9]

base_input_folder = 'data/two_neck_dataset/'
subfolders = ['parkin', 'normal']
input_file_name = '*.csv'
statistic = ['x_acc', 'y_acc', 'z_acc']



# # 各トレンドのデータ数
# n = 200
# # 変化点
# bkps = [200,400,600,800]
# # 各トレンドのデータを生成
# data1 = np.random.normal(0, 1, n)
# data2 = np.random.normal(0, 5, n)
# data3 = np.random.normal(0, 10, n)
# data4 = np.random.normal(0, 3, n)
# # サンプルデータ
# data = np.concatenate([data1,data2,data3,data4])
# signal_v = data.reshape(-1, 1)
# # プロット
# plt.plot(signal_v)
# plt.show()





for subfolder in subfolders:
    input_folder_path = os.path.join(base_input_folder, subfolder)
    csv_files = glob.glob(os.path.join(input_folder_path, input_file_name))

    features_by_symptom = find_change_point(csv_files)
    all_results.append(pd.DataFrame(features_by_symptom))

final_features = pd.concat(all_results)
save_features_to_csv(final_features, output_features_file)
