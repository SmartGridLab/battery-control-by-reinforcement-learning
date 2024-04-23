import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 日本語フォントの設定
mpl.rc('font', family='Meiryo')


# CSVファイルを読み込み
file_path = '/workspaces/battery-control-by-reinforcement-learning/Battery-Control-By-Reinforcement-Learning/result_dataframe.csv'
data = pd.read_csv(file_path)

# 時間データの生成
data['time'] = data['hour'].astype(float)

# 折れ線グラフの作成
fig, ax1 = plt.subplots(figsize=(12, 6))

# 左側の縦軸設定
ax1.plot(data['time'], data['PV_predict_bid[kW]'], 'r--', label='PV出力(予測)')
ax1.plot(data['time'], data['PV_actual[kW]'], 'r-', label='PV出力(実際)')
ax1.plot(data['time'], data['energyprice_predict_bid[Yen/kWh]'], 'g--', label='スポット市場価格(予測)')
ax1.plot(data['time'], data['energyprice_actual[Yen/kWh]'], 'g-', label='スポット市場価格(実際)')
ax1.set_xlabel('時間')
ax1.set_ylabel('PV出力[kW], 電力価格[円/kWh-30min]')
ax1.set_xticks(range(0, 25, 3))
ax1.set_yticks(range(0, 39, 2))
ax1.legend(loc='upper left')

# 右側の縦軸設定
ax2 = ax1.twinx()
ax2.plot(data['time'], data['SoC_bid[%]'], 'y--', label='SoC(予測)')
ax2.plot(data['time'], data['SoC_actual_bid[%]'], 'y-', label='SoC(実需給)')
ax2.set_ylabel('SoC[%]')
ax2.set_yticks(range(0, 101, 10))
ax2.legend(loc='upper right')


# x軸とy軸の補助目盛り線を非表示にする
ax1.xaxis.grid(False)
ax1.yaxis.grid(False)

# 右側の縦軸の補助目盛り線を非表示にする
ax2.yaxis.grid(False)

plt.title('充放電計画と動作結果')
plt.grid(True)
plt.savefig('result_graph.pdf', format='pdf')
plt.show()


