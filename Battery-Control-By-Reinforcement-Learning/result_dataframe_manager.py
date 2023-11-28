import pandas as pd

# 表示
print("dataframe作成")

# 列名をリストとして定義
columns = [
    'year', 'month', 'day', 'hour',
     
    #'PV_predict', 'PV_actual',
    'PV_predict', 'PV_predict_realtime', 'PV_actual',

    #'energyprice_predict', 'energyprice_actual', 'imbalanceprice_predict', 'imbalanceprice_actual',
    'energyprice_predict_bid', 'energyprice_predict_plan', 'energyprice_actual', 'imbalanceprice_predict_bid', 'imbalanceprice_predict_plan', 'imbalanceprice_actual',

    #'charge/discharge_bid', 'charge/discharge_plan', 'charge/discharge_actual',
    'charge/discharge_bid', 'charge/discharge_plan', 'charge/discharge_actual', 'charge/discharge_actual_realtime',

    #'SoC_bid', 'SoC_plan', 'SoC_actual', 'energytransfer_bid', 'energytransfer_plan', 'energytransfer_actual',
    'SoC_bid', 'SoC_plan', 'SoC_actual', 'SoC_actual_realtime', 'energytransfer_bid', 'energytransfer_plan', 'energytransfer_actual','energytransfer_actual_realtime',

    #'energy_profit', 'imbalance_penalty', 'total_profit',
    'energy_profit', 'energy_profit_realtime', 'imbalance_penalty', 'imbalance_penalty_realtime', 'total_profit', 'total_profit_realtime',

    'mode', 'mode_realtime'
]

# 空のDataframeを作成
dataframe = pd.DataFrame(columns=columns)


# DataframeをCSVファイルとして出力
# 作成時間も同時に記入できるとよさそう？
dataframe.to_csv('./Battery-Control-By-Reinforcement-Learning/result_dataframe.csv', index=False)