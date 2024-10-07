#最終的な取引について、収益を評価するプログラム
import pandas as pd

class ResultEvaluation:
    def __init__(self):
        # 全てのデータを読み込み
        self.df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        self.date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        # date_infoは {'year': year, 'month': month, 'day': day} の形式
        self.date_info['date'] = pd.to_datetime(self.date_info[['year', 'month', 'day']])
        self.latest_date = self.date_info['date'].max()
        # 現在日付を取得
        self.year = self.latest_date.year
        self.month = self.latest_date.month
        self.day = self.latest_date.day
        # 最新日付のデータをフィルタリング
        self.df = self.df_original[(self.df_original['year'] == self.year) & 
                                   (self.df_original['month'] == self.month) & 
                                   (self.df_original['day'] == self.day)]
        # year, month, day, hourのをindexとして設定
        self.df_original.set_index(['year', 'month', 'day', 'hour'], inplace = True)
        self.df.set_index(['year', 'month', 'day', 'hour'], inplace = True)

    def evaluation_bid_result(self):
        # _actual = 計画したものを実際に実行した場合の数値
        # _actualなし = ただ予測し計画した段階の計画値

        # 'energy_transfer'列を計算
        self.df["energytransfer_bid[kWh]"] = self.df["PV_predict_bid[kW]"] * 0.5 + self.df["charge/discharge_bid[kWh]"]
        self.df["energytransfer_actual_bid[kWh]"] = self.df["PV_actual[kW]"] * 0.5 + self.df["charge/discharge_actual_bid[kWh]"]
        # "energy_profit" 列を計算
        self.df["energyprofit_bid[Yen]"] = self.df["energyprice_actual[Yen/kWh]"] * self.df["energytransfer_actual_bid[kWh]"]
        # imbalancepenalty_actual_bid = | bid(１日前)で計画した売電計画量 - bidで予測した売電計画量で実行したときの売電量 | * (-1)実際のインバランス料金
        self.df["imbalancepenalty_actual_bid[Yen]"] = abs(self.df["energytransfer_bid[kWh]"] - self.df["energytransfer_actual_bid[kWh]"]) * self.df["imbalanceprice_actual[Yen/kWh]"] * (-1)
        # "total_profit" 列を計算
        # totalprofit_bid: bidの段階ではimbalanceが発生しないので、energyprofit_bidがそのままtotalprofit_bidになる
        self.df["totalprofit_bid[Yen]"] = self.df["energyprofit_bid[Yen]"]
        self.df["totalprofit_actual_bid[Yen]"] = self.df["energyprofit_bid[Yen]"] + self.df["imbalancepenalty_actual_bid[Yen]"]
        
    def evaluation_realtime_result(self):
        # _actual = 計画したものを実際に実行した場合の数値
        # _actualなし = ただ予測し計画した段階の計画値
        
        # "energy_transfer" 列を計算
        self.df["energytransfer_realtime[kWh]"] = self.df["PV_predict_realtime[kW]"] * 0.5 + self.df["charge/discharge_realtime[kWh]"]
        self.df["energytransfer_actual_realtime[kWh]"] = self.df["PV_actual[kW]"] * 0.5 + self.df["charge/discharge_actual_realtime[kWh]"]
        # "energy_profit" 列を計算
        self.df["energyprofit_realtime[Yen]"] = self.df["energyprice_actual[Yen/kWh]"] * self.df["energytransfer_actual_realtime[kWh]"]
        # imbalancepenalty_actual_realtime = | bid(１日前)で計画した売電量 - realtimeで予測した売電計画量で実行したときの実際売電量 | * (-1)実際のインバランス料金
        self.df["imbalancepenalty_actual_realtime[Yen]"] = abs(self.df["energytransfer_bid[kWh]"] - self.df["energytransfer_actual_realtime[kWh]"]) * self.df["imbalanceprice_actual[Yen/kWh]"] * (-1)
        # imbalancepenalty_realtime = | bid(1日前)で計画した売電量 - realtime(30分前)で計画した売電量 |* (-1) realtimeで予測したインバランス料金
        self.df["imbalancepenalty_realtime[Yen]"] = abs(self.df["energytransfer_bid[kWh]"] - self.df["energytransfer_realtime[kWh]"]) * self.df["imbalanceprice_predict_realtime[Yen/kWh]"] * (-1)
        # "total_profit" 列を計算
        # totalprofit_realtime: realtimeの場合は、imbalancepenalty_realtimeが存在している
        self.df["totalprofit_realtime[Yen]"] = self.df["energyprofit_realtime[Yen]"] + self.df["imbalancepenalty_realtime[Yen]"]
        self.df["totalprofit_actual_realtime[Yen]"] = self.df["energyprofit_realtime[Yen]"] + self.df["imbalancepenalty_actual_realtime[Yen]"]
    
    def evaluation_result_save(self,mode):
        print("\n---動作結果評価開始---")
        if mode == "bid":
            self.evaluation_bid_result()
        elif mode == "realtime":
            self.evaluation_realtime_result()

        self.df_original.update(self.df)
        # indexを振りなおす
        self.df_original.reset_index(inplace = True)
        self.df.reset_index(inplace = True)
        self.df_original.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", header = True, index=False)
        print("---実動作結果評価終了---")