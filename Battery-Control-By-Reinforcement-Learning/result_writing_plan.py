import pandas as pd

print("-realtimeモード結果書き込み開始-")

# result_data.csvを読み込む
result_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_data.csv")

# 列名の変更
result_data.rename(columns={
    "PVout": "PV_predict",
    "price": "energyprice_predict",
    "imbalance": "imbalanceprice_predict",
    "charge/discharge": "charge/discharge_plan",
    "SoC": "SoC_plan",
    "energy_transfer": "energytransfer_plan"
}, inplace=True)

# "hour"列で23.5が格納されている行のインデックスを取得
index_to_keep = result_data[result_data['hour'] == 23.5].index

# 23.5の行までを残し、それ以降の行を削除
result_data = result_data.loc[:index_to_keep.max()]

# result_dataframe.csvを読み込む
existing_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

# ###ここから変更
# 0行目のhourの値がexisting_dataのどの行に対応するかを探索
x_row = existing_data[existing_data['hour'] == result_data['hour'].iloc[0]].index[0]

# result_dataから必要な列を取得し、existing_dataの対応する行に格納
existing_data.loc[x_row:47, 'PV_predict'] = result_data['PV_predict'].values
existing_data.loc[x_row:47, 'energyprice_predict'] = result_data['energyprice_predict'].values
existing_data.loc[x_row:47, 'imbalanceprice_predict'] = result_data['imbalanceprice_predict'].values
existing_data.loc[x_row:47, 'charge/discharge_plan'] = result_data['charge/discharge_plan'].values
existing_data.loc[x_row:47, 'SoC_plan'] = result_data['SoC_plan'].values
existing_data.loc[x_row:47, 'energytransfer_plan'] = result_data['energytransfer_plan'].values

# 更新されたデータをresult_dataframe.csvに保存
existing_data.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)

print("-realtimeモード結果書き込み完了-")

