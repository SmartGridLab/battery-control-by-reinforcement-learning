import pandas as pd
import numpy as np

battery_MAX = 4

result_dataframe = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

# 蓄電池データ取得
# dataframe最終行から取得
last_soc_actual_row = result_dataframe[result_dataframe['SoC_actual'].notna()].tail(1)

if not last_soc_actual_row.empty:
    last_soc_actual_value = last_soc_actual_row['SoC_actual'].values[0]
    now_battery = last_soc_actual_value * 0.01 * battery_MAX  # 電力量[kWh]に変換
else:
    now_battery = 0

print(now_battery)

print("--テスト--")
