#最終的な取引について、収益を評価するプログラム
import pandas as pd

# CSVファイルを読み込み
df = pd.read_csv("result_dataframe.csv")

# "energy_profit" 列を計算
df["energy_profit"] = df["energyprice_actual"] * df["energytransfer_actual"]

# "imbalanceprice_actual" 列を計算
df["imbalanceprice_actual"] = (df["energytransfer_actual"] - df["energytransfer_bid"]) * df["imbalanceprice_actual"]

# "total_profit" 列を計算
df["total_profit"] = df["energy_profit"] - df["imbalanceprice_actual"]

# 計算結果をCSVファイルに上書き保存
df.to_csv("result_dataframe.csv", index=False)
