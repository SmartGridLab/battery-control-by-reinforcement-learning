import pandas as pd

print("-bidモード結果書き込み開始-")

# result_data.csvを読み込む
result_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_data.csv")

# 列名の変更
result_data.rename(columns={
    "PVout": "PV_predict",
    "price": "energyprice_predict",
    "imbalance": "imbalanceprice_predict",
    "charge/discharge": "charge/discharge_bid",
    "SoC": "SoC_bid",
    "energy_transfer": "energytransfer_bid"
}, inplace=True)

# result_dataframe.csvを読み込む
existing_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

# 同じ名前の行に追記する
updated_data = pd.concat([existing_data, result_data], ignore_index=True)

# 余計なインデックス番号の列を消去
updated_data.drop(columns=["Unnamed: 0"], inplace=True)

# 更新されたデータをresult_dataframe.csvに保存
updated_data.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)

print("-bidモード結果書き込み完了-")
