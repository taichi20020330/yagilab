import numpy as np
import ruptures as rpt
from ruptures.metrics import precision_recall
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 9]

# 各トレンドのデータ数
n = 200
# 変化点
bkps = [200,400,600,800]
# 各トレンドのデータを生成
data1 = np.random.normal(0, 1, n)
data2 = np.random.normal(0, 5, n)
data3 = np.random.normal(0, 10, n)
data4 = np.random.normal(0, 3, n)
# サンプルデータ
data = np.concatenate([data1,data2,data3,data4])
signal_v = data.reshape(-1, 1)
# プロット
plt.plot(signal_v)
plt.show()

# コスト関数の設定
model = "normal"
# アルゴの設定と学習
algo = rpt.Pelt(model=model).fit(signal_v)
# 変化点の検出
my_bkps = algo.predict(n_bkps=3)
# plot
rpt.display(signal_v, bkps, my_bkps, figsize=(12, 9))
plt.show()
# 検出された変化点
print(my_bkps)