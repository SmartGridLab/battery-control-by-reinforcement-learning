#最終的な取引について、収益を評価するプログラム
import pandas as pd

print("\n---動作結果評価開始---")

# CSVファイルを読み込み
df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

# "energy_profit" 列を計算
df["energy_profit"] = df["energyprice_actual"] * df["energytransfer_actual"]

# "imbalance_penalty" 列を計算
df["imbalance_penalty"] = abs(df["energytransfer_actual"] - df["energytransfer_bid"]) * df["imbalanceprice_actual"]

# "total_profit" 列を計算
df["total_profit"] = df["energy_profit"] - df["imbalance_penalty"]

# 計算結果をCSVファイルに上書き保存
df.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)

print("---実動作結果評価終了---")
