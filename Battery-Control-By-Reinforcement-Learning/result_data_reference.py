import pandas as pd

def main():
    print("-実績データ参照開始-")

    # result_data.csvからyear, month, dayを取得
    result_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_data.csv")
    year = int(result_data.loc[0, "year"])
    month = int(result_data.loc[0, "month"])
    day = int(result_data.loc[0, "day"])
    
    # 既存のCSVファイルを読み込む
    dataframe = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
    
    # データを取得
    data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
    PV_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["PVout"]]
    energyprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["price",]]
    imbalanceprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["imbalance"]]

    # pd -> numpy変換(pandasのままだとindex名が残り面倒)
    PV_actual = PV_actual.values
    energyprice_actual = energyprice_actual.values
    imbalanceprice_actual = imbalanceprice_actual.values


    dataframe['PV_actual'] = PV_actual
    dataframe['energyprice_actual'] = energyprice_actual
    dataframe['imbalanceprice_actual'] = imbalanceprice_actual
   
    # 結果を上書き保存
    dataframe.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)

    print("-実績データ書き込み完了-")

if __name__ == "__main__":
    main()

