import pandas as pd

# 表示
print("dataframe作成")

# 列名をリストとして定義
columns = [
    'year', 'month', 'day', 'hour', 'PV_predict', 'PV_actual',
    'energyprice_predict', 'energyprice_actual', 'imbalanceprice_predict', 'imbalanceprice_actual',
    'charge/discharge_bid', 'charge/discharge_plan', 'charge/discharge_actual',
    'SoC_bid', 'SoC_plan', 'SoC_actual', 'energytransfer_bid', 'energytransfer_plan', 'energytransfer_actual',
    'energy_profit', 'imbalance_penalty', 'total_profit',
    'mode'
]

# 空のDataframeを作成
dataframe = pd.DataFrame(columns=columns)


# DataframeをCSVファイルとして出力
# 作成時間も同時に記入できるとよさそう？
dataframe.to_csv('./Battery-Control-By-Reinforcement-Learning/result_dataframe.csv', index=False)