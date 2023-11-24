import pandas as pd

def main():
    print("---実績データ参照開始---")

    # result_data.csvからyear, month, dayを取得
    result_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_data.csv")
    year = int(result_data.loc[0, "year"])
    month = int(result_data.loc[0, "month"])
    day = int(result_data.loc[0, "day"])
    
    # result_dataframe.csvを読み込む
    dataframe = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
    
    # 学習データから指定した日付のPV, price, imbalance実績値を取得
    data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
    PV_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["PVout"]]
    energyprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["price",]]
    imbalanceprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["imbalance"]]

    # pd -> numpy変換
    PV_actual = PV_actual.values.flatten()
    energyprice_actual = energyprice_actual.values.flatten()
    imbalanceprice_actual = imbalanceprice_actual.values.flatten()


    ### 実績データの挿入
    # 実績データが格納されている最後の行番号を取得(何行目からデータを挿入するか判定するため)
    last_data_row = dataframe['PV_actual'].last_valid_index()
    i = last_data_row
    # 0行目から格納するとき(1日目のデータを入れる場合はdataframeにまだデータがない)
    if i == None:
        i = -1
    # i+1行目からi+48行目に実績データを挿入
    dataframe.iloc[i+1:i+49, dataframe.columns.get_loc('PV_actual')] = PV_actual
    dataframe.iloc[i+1:i+49, dataframe.columns.get_loc('energyprice_actual')] = energyprice_actual * 0.5
    dataframe.iloc[i+1:i+49, dataframe.columns.get_loc('imbalanceprice_actual')] = imbalanceprice_actual * 0.5
   
    # result_dataframe.csvを上書き保存
    dataframe.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)

    print("---実績データ書き込み完了---")

if __name__ == "__main__":
    main()

