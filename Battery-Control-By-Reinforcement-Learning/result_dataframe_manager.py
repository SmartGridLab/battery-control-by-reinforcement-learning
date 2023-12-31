import pandas as pd

# 表示
print("dataframe作成")

# 列名をリストとして定義
columns = [
    # 時系列関係
    'year', 'month', 'day', 'hour',
    # PV関連
    'PV_predict', 'PV_predict_realtime', 'PV_actual',
    # 電力価格関連
    'energyprice_predict_bid', 'energyprice_predict_plan', 'energyprice_actual', 'imbalanceprice_predict_bid', 'imbalanceprice_predict_plan', 'imbalanceprice_actual',
    # 充放電量関連
    'charge/discharge_bid', 'charge/discharge_plan', 'charge/discharge_actual', 'charge/discharge_actual_realtime',
    # SoC関連
    'SoC_bid', 'SoC_plan', 'SoC_actual', 'SoC_actual_realtime', 
    # 売電量関連
    'energytransfer_bid', 'energytransfer_realtime', 'energytransfer_bid_actual','energytransfer_realtime_actual',
    # 売電利益関連
    'energy_profit_bid', 'energy_profit_realtime', 'imbalance_penalty_bid', 'imbalance_penalty_realtime', 'total_profit_bid', 'total_profit_realtime',
    # 動作モード：operateの条件分岐を見るためのもの。デバッグ用
    'mode', 'mode_realtime'
]

# 空のDataframeを作成
dataframe = pd.DataFrame(columns=columns)


# DataframeをCSVファイルとして出力
# 作成時間も同時に記入できるとよさそう？
dataframe.to_csv('./Battery-Control-By-Reinforcement-Learning/result_dataframe.csv', index=False)