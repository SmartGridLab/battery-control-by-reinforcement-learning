import pandas as pd

# 表示
print("dataframe作成")

# 列名をリストとして定義
columns = [
    # 時系列関係
    'year', 'month', 'day', 'hour',
    # PV関連
    'PV_predict_bid', 'PV_predict_realtime', 'PV_actual',
    # 電力価格関連
    'energyprice_predict_bid', 'energyprice_predict_realtime', 'energyprice_actual', 
    'imbalanceprice_predict_bid', 'imbalanceprice_predict_realtime', 'imbalanceprice_actual',
    # 充放電量関連
    'charge/discharge_bid', 'charge/discharge_realtime', 'charge/discharge_actual_bid', 'charge/discharge_actual_realtime',
    # SoC関連
    'SoC_bid', 'SoC_realtime', 'SoC_actual_bid', 'SoC_actual_realtime', 
    # 売電量関連
    'energytransfer_bid', 'energytransfer_realtime', 'energytransfer_actual_bid','energytransfer_actual_realtime',
    # 売電利益関連
    'energyprofit_bid', 'energyprofit_realtime', 'imbalancepenalty_bid', 'imbalancepenalty_realtime', 'totalprofit_bid', 'totalprofit_realtime',
    # 動作モード：operateの条件分岐を見るためのもの。デバッグ用
    'mode', 'mode_realtime'
]

# 空のDataframeを作成
dataframe = pd.DataFrame(columns=columns)

# DataframeをCSVファイルとして出力
# 作成時間も同時に記入できるとよさそう？
dataframe.to_csv('./Battery-Control-By-Reinforcement-Learning/result_dataframe.csv', index=False)