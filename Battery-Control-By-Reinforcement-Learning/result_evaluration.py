#最終的な取引について、収益を評価するプログラム
import pandas as pd

def main():
    print("\n---動作結果評価開始---")

    date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
            # date_infoは {'year': year, 'month': month, 'day': day} の形式
    date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
    latest_date = date_info['date'].max()

    year = latest_date.year
    month = latest_date.month
    day = latest_date.day

     # 全てのデータを読み込み
    original_df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

    # 最新日付のデータをフィルタリング
    df = original_df[(original_df['year'] == year) & 
                     (original_df['month'] == month) & 
                     (original_df['day'] == day)]

    # 'energy_transfer'列を計算
    df["energytransfer_bid[kWh]"] = df["PV_predict_bid[kW]"] * 0.5 + df["charge/discharge_bid[kWh]"]
    df["energytransfer_actual_bid[kWh]"] = df["PV_actual[kW]"] * 0.5 + df["charge/discharge_actual_bid[kWh]"]

    # "energy_profit" 列を計算
    df["energyprofit_bid[Yen]"] = df["energyprice_actual[Yen/kWh]"] * df["energytransfer_actual_bid[kWh]"]
    # df["energy_profit_realtime"] = df["energyprice_actual"] * df["energytransfer_actual_realtime"]

    # "imbalance_penalty" 列を計算
    df["imbalancepenalty_actual_bid[Yen]"] = abs(df["energytransfer_actual_bid[kWh]"] - df["energytransfer_bid[kWh]"]) * df["imbalanceprice_actual[Yen/kWh]"] * (-1)
    # df["imbalancepenalty_actual_realtime"] = abs(df["energytransfer_actual_realtime"] - df["energytransfer_bid"]) * df["imbalanceprice_actual"] * (-1)

    # "total_profit" 列を計算
    # totalprofit_bid: bidの段階ではimbalanceが発生しないので、energyprofit_bidがそのままtotalprofit_bidになる
    # total_profit_realtime: realtimeの場合は、imbalancepenalty_realtimeが存在している可能性がある。要検討。
    df["totalprofit_bid[Yen]"] = df["energyprofit_bid[Yen]"]
    df["totalprofit_actual_bid[Yen]"] = df["energyprofit_bid[Yen]"] + df["imbalancepenalty_actual_bid[Yen]"]
    # df["total_profit_realtime"] = df["energyprofit_realtime"] + df["imbalancepenalty_realtime"]
    # df["totalprofit_actual_realtime"] = df["energyprofit_realtime"] + df["imbalancepenalty_actual_realtime"]


    # フィルタリングした部分のデータを元データから消す
    original_df_erase = original_df[~((original_df['year'] == year) & 
                             (original_df['month'] == month) & 
                             (original_df['day'] == day))]
    
    # 元のデータフレームに追加
    original_df_concat = pd.concat([original_df_erase, df], axis=0)

    print(df)
    print(original_df_erase)
    print(original_df_concat)
    # 計算結果をCSVファイルに上書き保存
    original_df_concat.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)

    print("---実動作結果評価終了---")

if __name__ == "__main__":
    main()