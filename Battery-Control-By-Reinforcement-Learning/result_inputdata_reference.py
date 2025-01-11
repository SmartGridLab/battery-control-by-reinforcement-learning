# 充放電計画の性能評価のためのデータを集めるコード
# - PV発電量の実績値、電力価格の実績値、不平衡電力価格の実績値を取得する
# - 実績値ベースでの売電による収益の計算を行う

import pandas as pd
from RL_test import TestModel

class ResultInputDataReference:
    def __init__(self):
        self.actual_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/train_data/input_data2022_0.csv")
        # 現在の日付を取得
        self.year, self.month, self.day = self.get_current_date()
        # 実績データの取得
        self.PV_actual = self.actual_data[(self.actual_data["year"] == self.year) & (self.actual_data["month"] == self.month) & (self.actual_data["day"] == self.day)][["PVout"]]
        self.energyprice_actual = self.actual_data[(self.actual_data["year"] == self.year) & (self.actual_data["month"] == self.month) & (self.actual_data["day"] == self.day)][["price"]]
        self.imbalanceprice_actual = self.actual_data[(self.actual_data["year"] == self.year) & (self.actual_data["month"] == self.month) & (self.actual_data["day"] == self.day)][["imbalance"]]

        # pd -> numpy変換(2次元配列 → 1次元配列)
        ## ------------------テストデータをここで作る----------------------------
        # 0.5掛ける必要あるのか？
        self.PV_actual = self.PV_actual.values.flatten()
        self.energyprice_actual = self.energyprice_actual.values.flatten() * 0.5
        self.imbalanceprice_actual = self.imbalanceprice_actual.values.flatten() * 0.5
        ## -------------------------------------------------------------------
    
        # 現在の日付を取得
    def get_current_date(self):
        date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        # date_infoは {'year': year, 'month': month, 'day': day, 'hour': hour} の形式
        date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
        latest_date = date_info['date'].max()
        year = latest_date.year
        month = latest_date.month
        day = latest_date.day
        return year, month, day
    
    # 新しいDataFrameを作成 & 実測値の取得
    def make_actual_dataframe(self):
        print("\n---新しい日付データフレームの作成 & 実測値の取得---")
        df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        predictdate_exists = ((df_original["year"] == self.year) &
                          (df_original["month"] == self.month) &
                          (df_original["day"] == self.day)).any()
        # 同じ日付データが存在する場合
        if predictdate_exists:
            print("\n---すでに存在する日付データフレームのためスキップ---")
            pass
        # 存在しない場合
        else:
            print("\n---新しい日付のため実測データを取得---")
            df_new = pd.DataFrame(index = range(48), columns = df_original.columns)
            df_new["year"], df_new["month"], df_new["day"] = self.year, self.month, self.day
            df_new["hour"] = [i * 0.5 for i in range(48)]
            # 実測値を挿入
            # ↓ 0.5倍する？ ↓
            df_new.loc[df_new.index[:48], 'PV_actual[kW]'] = self.PV_actual
            df_new.loc[df_new.index[:48], 'energyprice_actual[Yen/kWh]'] = self.energyprice_actual
            df_new.loc[df_new.index[:48], 'imbalanceprice_actual[Yen/kWh]'] = self.imbalanceprice_actual
            # 下からデータフレームを追加
            df_new.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", mode = "a", header = False, index = False)
            print("\n---実測データ取得完了---")

    def input_actualdata(self, mode):
        print("---実績データ参照開始---")
        # 現在日付を取得
        date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        # date_infoは {'year': year, 'month': month, 'day': day, 'hour': hour} の形式
        date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
        latest_date = date_info['date'].max()
        year = latest_date.year
        month = latest_date.month
        day = latest_date.day
        # 全てのデータを読み込み
        df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        # 現在日付のデータをフィルタリング
        df = df_original[(df_original['year'] == year) & 
                        (df_original['month'] == month) & 
                        (df_original['day'] == day)].reset_index(drop = True)
        # 実績データがすでに存在するか確認（存在する ＝ True, 存在しない = False）
        actualdata_exists = ((df['year'] == year) & 
                            (df['month'] == month) & 
                            (df['day'] == day) &
                            pd.notna(df['PV_actual[kW]']) &
                            pd.notna(df['energyprice_actual[Yen/kWh]']) &
                            pd.notna(df['imbalanceprice_actual[Yen/kWh]'])).any()
        if actualdata_exists:
            print("---実績データがすでに存在します---")
        else:
            print("---実績データが存在しないため、実績データを入力します---")
            data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
            PV_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["PVout"]]
            energyprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["price"]]
            imbalanceprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["imbalance"]]
            # pd -> numpy変換(2次元配列 → 1次元配列)
            PV_actual = PV_actual.values.flatten()
            energyprice_actual = energyprice_actual.values.flatten()
            imbalanceprice_actual = imbalanceprice_actual.values.flatten()
            # 実績データの挿入
            # -------------------------- debug --------------------------
            # if mode == "realtime":
            #     df.loc[df.index[:48], 'PV_actual[kW]'] = PV_actual
            # elif mode == "bid":
            #     df.loc[df.index[:48], 'PV_actual[kW]'] = PV_actual * 10
            # -------------------------- debug --------------------------
            df.loc[df.index[:48], 'PV_actual[kW]'] = PV_actual
            df.loc[df.index[:48], 'energyprice_actual[Yen/kWh]'] = energyprice_actual
            df.loc[df.index[:48], 'imbalanceprice_actual[Yen/kWh]'] = imbalanceprice_actual
            # year, month, day, hourをindexとして設定
            df_original.set_index(['year', 'month', 'day', 'hour'], inplace = True)
            df.set_index(['year', 'month', 'day', 'hour'], inplace = True)
            df_original.update(df)
            # indexを振りなおす
            df_original.reset_index(inplace = True)
            df.reset_index(inplace = True)
            df_original.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", header = True, index=False)
            print("---実績データの入力完了---")
