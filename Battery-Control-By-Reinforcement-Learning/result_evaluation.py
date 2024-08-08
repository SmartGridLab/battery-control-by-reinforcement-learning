#最終的な取引について、収益を評価するプログラム
import pandas as pd

class ResultEvaluation:
    def __init__(self):
        self.date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        # date_infoは {'year': year, 'month': month, 'day': day} の形式
        self.date_info['date'] = pd.to_datetime(self.date_info[['year', 'month', 'day']])
        self.latest_date = self.date_info['date'].max()

        self.year = self.latest_date.year
        self.month = self.latest_date.month
        self.day = self.latest_date.day
        # 全てのデータを読み込み
        self.original_df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        # 最新日付のデータをフィルタリング
        self.df = self.original_df[(self.original_df['year'] == self.year) & 
                        (self.original_df['month'] == self.month) & 
                        (self.original_df['day'] == self.day)]

    def evaluation_bid_result(self):
        # 'energy_transfer'列を計算
        self.df["energytransfer_bid[kWh]"] = self.df["PV_predict_bid[kW]"] * 0.5 + self.df["charge/discharge_bid[kWh]"]
        self.df["energytransfer_actual_bid[kWh]"] = self.df["PV_actual[kW]"] * 0.5 + self.df["charge/discharge_actual_bid[kWh]"]
        # "energy_profit" 列を計算
        self.df["energyprofit_bid[Yen]"] = self.df["energyprice_actual[Yen/kWh]"] * self.df["energytransfer_actual_bid[kWh]"]
        # "imbalance_penalty" 列を計算
        self.df["imbalancepenalty_actual_bid[Yen]"] = abs(self.df["energytransfer_actual_bid[kWh]"] - self.df["energytransfer_bid[kWh]"]) * self.df["imbalanceprice_actual[Yen/kWh]"] * (-1)
        
        # "total_profit" 列を計算
        # totalprofit_bid: bidの段階ではimbalanceが発生しないので、energyprofit_bidがそのままtotalprofit_bidになる
        self.df["totalprofit_bid[Yen]"] = self.df["energyprofit_bid[Yen]"]
        self.df["totalprofit_actual_bid[Yen]"] = self.df["energyprofit_bid[Yen]"] + self.df["imbalancepenalty_actual_bid[Yen]"]
        
    def evaluation_realtime_result(self):
        # "energy_transfer" 列を計算
        self.df["energytransfer_realtime[kWh]"] = self.df["PV_predict_realtime[kW]"] * 0.5 + self.df["charge/discharge_realtime[kWh]"]
        self.df["energytransfer_actual_realtime[kWh]"] = self.df["PV_actual[kW]"] * 0.5 + self.df["charge/discharge_actual_realtime[kWh]"]
        # "energy_profit" 列を計算
        self.df["energyprofit_realtime[Yen]"] = self.df["energyprice_actual[Yen/kWh]"] * self.df["energytransfer_actual_realtime[kWh]"]
        # "imbalance_penalty" 列を計算
        self.df["imbalancepenalty_actual_realtime[Yen]"] = abs(self.df["energytransfer_actual_realtime[kWh]"] - self.df["energytransfer_realtime[kWh]"]) * self.df["imbalanceprice_actual[Yen/kWh]"] * (-1)

        # "total_profit" 列を計算
        # totalprofit_realtime: realtimeの場合は、imbalancepenalty_realtimeが存在している可能性がある。要検討。
        self.df["total_profit_realtime[Yen]"] = self.df["energyprofit_realtime[Yen]"] + self.df["imbalancepenalty_realtime[Yen]"]
        self.df["totalprofit_actual_realtime[Yen]"] = self.df["energyprofit_realtime[Yen]"] + self.df["imbalancepenalty_actual_realtime[Yen]"]

    def evaluation_result_save(self,mode):
        print("\n---動作結果評価開始---")
        if mode == "bid":
            self.evaluation_bid_result()
        elif mode == "realtime":
            self.evaluation_realtime_result()

         # フィルタリングした部分のデータを元データから消す
        original_df_erase = self.original_df[~((self.original_df['year'] == self.year) & 
                                (self.original_df['month'] == self.month) & 
                                (self.original_df['day'] == self.day))]
        # 元のデータフレームに追加
        original_df_concat = pd.concat([original_df_erase, self.df], axis=0)
        print(self.df)
        print(original_df_erase)
        print(original_df_concat)
        # 計算結果をCSVファイルに上書き保存
        original_df_concat.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)
        print("---実動作結果評価終了---")