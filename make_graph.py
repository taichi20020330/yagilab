import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


base_input_folder = 'data/two_neck_dataset/'
base_output_folder = 'graph'
subfolders = ['parkin', 'normal']
output_folder_name =  'graph'
input_file_name = '*.csv'
statistic = ['x_acc', 'y_acc', 'z_acc']


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
    # plt.show()

    save_graph_to_png(plt, output_file_path)


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
        df = pd.read_csv(file, header=None, names=['x_acc', 'y_acc', 'z_acc'])
        make_graph(df, output_file_path)


