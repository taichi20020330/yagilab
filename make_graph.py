import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルのパスを指定
csv_file_path = 'your_file_path.csv'

# CSVファイルを読み込み、列名を設定
df = pd.read_csv(csv_file_path, sep='\t', header=None, names=['x_acc', 'y_acc', 'z_acc'])

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

# グラフを表示
plt.show()
