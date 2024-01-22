#最終的な取引について、収益を評価するプログラム
import pandas as pd

print("\n---動作結果評価開始---")

# CSVファイルを読み込み
df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

# 'energy_transfer'列を計算
df["energytransfer_bid"] = df["PV_predict_bid"] + df["charge/discharge_bid"]
df["energytransfer_actual_bid"] = df["PV_actual"] + df["charge/discharge_actual_bid"]

# "energy_profit" 列を計算
df["energyprofit_bid"] = df["energyprice_actual"] * df["energytransfer_actual_bid"]
# df["energy_profit_realtime"] = df["energyprice_actual"] * df["energytransfer_actual_realtime"]

# "imbalance_penalty" 列を計算
df["imbalancepenalty_actual_bid"] = abs(df["energytransfer_actual_bid"] - df["energytransfer_bid"]) * df["imbalanceprice_actual"] * (-1)
# df["imbalancepenalty_actual_realtime"] = abs(df["energytransfer_actual_realtime"] - df["energytransfer_bid"]) * df["imbalanceprice_actual"] * (-1)

# "total_profit" 列を計算
# totalprofit_bid: bidの段階ではimbalanceが発生しないので、energyprofit_bidがそのままtotalprofit_bidになる
# total_profit_realtime: realtimeの場合は、imbalancepenalty_realtimeが存在している可能性がある。要検討。
df["totalprofit_bid"] = df["energyprofit_bid"]
df["totalprofit_actual_bid"] = df["energyprofit_bid"] + df["imbalancepenalty_actual_bid"]
# df["total_profit_realtime"] = df["energyprofit_realtime"] + df["imbalancepenalty_realtime"]
# df["totalprofit_actual_realtime"] = df["energyprofit_realtime"] + df["imbalancepenalty_actual_realtime"]

# 計算結果をCSVファイルに上書き保存
df.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)

print("---実動作結果評価終了---")
