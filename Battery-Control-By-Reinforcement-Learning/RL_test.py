# 外部モジュール
import warnings
import pandas as pd
import math
import numpy as np
from stable_baselines3 import PPO
import numpy as np

# 内製モジュール
from RL_env import ESS_ModelEnv as Env_bid
from RL_env_realtime import ESS_ModelEnv as Env_realtime
from RL_dataframe_manager import Dataframe_Manager as Dfmanager
from RL_operate import Battery_operate as Operate

warnings.simplefilter('ignore')
# 学習済みの強化学習モデルのテスト（評価）を行うクラス
class TestModel():
    def __init__(self, mode):

        self.dfmanager = Dfmanager() # データ読込みクラスのインスタンス化
        # 正規化前の元データを保持
        self.df_test_original = getattr(self.dfmanager, f"get_test_df_{mode}")()
        self.df_test = self.df_test_original.copy()
        print(f"self.df_test: {self.df_test}")
        self.boundary_soc_df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/for_debug/boundary_soc.csv")

        # edited_action_listを初期化
        self.edited_action_list = []
        self.natural_action_list = []
        self.energytransfer_list = []
        self.action_debug_list = []

        # クラスのインスタンス化
        if mode == "bid":
            self.env_bid = Env_bid(mode)
            # 正規化パラメータの取得
            self.pvout_max = self.env_bid.pvout_max
            self.pvout_min = self.env_bid.pvout_min
            self.price_max = self.env_bid.price_max
            self.price_min = self.env_bid.price_min
            self.imbalance_max = self.env_bid.imbalance_max
            self.imbalance_min = self.env_bid.imbalance_min
            self.soc_list = [self.boundary_soc_df["Initial_SoC_actual_bid"][0]] # その日のSoCの初期値 = 前日のSoC終値
        elif mode == "realtime":
            self.env_realtime = Env_realtime(mode)
            # 正規化パラメータの取得
            self.pvout_max = self.env_realtime.pvout_max
            self.pvout_min = self.env_realtime.pvout_min
            self.price_max = self.env_realtime.price_max
            self.price_min = self.env_realtime.price_min
            self.imbalance_max = self.env_realtime.imbalance_max
            self.imbalance_min = self.env_realtime.imbalance_min
            self.soc_list = [self.boundary_soc_df["Initial_SoC_actual_realtime"][0]] # その日のSoCの初期値 = 前日のSoC終値

        # テストデータの正規化
        if mode == "bid":
            self.df_test[f'PV_predict_{mode}_normalized'] = self.env_bid.normalize(self.df_test[f'PV_predict_{mode}[kW]'], self.pvout_max, self.pvout_min)
            self.df_test[f'energyprice_predict_{mode}_normalized'] = self.env_bid.normalize(self.df_test[f'energyprice_predict_{mode}[Yen/kWh]'], self.price_max, self.price_min)
            self.df_test[f'imbalanceprice_predict_{mode}_normalized'] = self.env_bid.normalize(self.df_test[f'imbalanceprice_predict_{mode}[Yen/kWh]'], self.imbalance_max, self.imbalance_min)

        elif mode == "realtime":
            self.df_test[f"PV_predict_{mode}_normalized"] = self.env_realtime.normalize(self.df_test[f"PV_predict_{mode}[kW]"], (self.env_realtime.upper_times * self.pvout_max), (self.env_realtime.lower_times * self.pvout_min))
            self.df_test[f"energyprice_predict_{mode}_normalized"] = self.env_realtime.normalize(self.df_test[f"energyprice_predict_{mode}[Yen/kWh]"], (self.env_realtime.upper_times * self.price_max), (self.env_realtime.lower_times * self.price_min))
            self.df_test[f"imbalanceprice_predict_{mode}_normalized"] = self.env_realtime.normalize(self.df_test[f"imbalanceprice_predict_{mode}[Yen/kWh]"], (self.env_realtime.upper_times * self.imbalance_max), (self.env_realtime.lower_times * self.imbalance_min))
            # 前日入札値の正規化
            self.df_test[f"energytransfer_bid_normalized"] = self.env_realtime.normalize(self.df_test["energytransfer_bid[kWh]"], (self.pvout_max * (1 + self.env_realtime.upper_times) + (self.env_realtime.upper_times * 0.5)), 0)


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

    # 学習したモデルをテスト
    def dispatch_test_bid(self, model_name, mode):
        # 学習済みモデルのロード
        self.model = PPO.load(model_name)

        # テストデータの各ステップについてループ
        for idx_state in range(len(self.df_test)):
            # 現在のタイムステップを取得
            time_of_day = idx_state % self.env_bid.day_steps
            theta = 2 * math.pi * time_of_day / self.env_bid.day_steps
            sin_time = math.sin(theta)
            cos_time = math.cos(theta)

            # 現在のSoCを取得
            current_soc = self.soc_list[-1] # 0.0 ~ 1.0[割合]
            PV_predict = self.df_test.at[idx_state, f"PV_predict_{mode}[kW]"]
            # 前日の入札値を取得

            # 観測値の構築
            obs = [
                self.df_test.at[idx_state, f"PV_predict_{mode}_normalized"], # [正規化]
                self.df_test.at[idx_state, f"energyprice_predict_{mode}_normalized"], # [正規化]
                self.df_test.at[idx_state, f"imbalanceprice_predict_{mode}_normalized"],# [正規化]
                current_soc,  # SoC
                sin_time, # [正規化]
                cos_time # [正規化]
            ]

            # モデルによるアクションの予測
            action, _states = self.model.predict(obs, deterministic=True)

            # for debug
            self.action_debug_list.append(action)

            edited_action, next_soc, energytransfer = Operate().operate_plan(PV_predict, action[0], current_soc)
            # アクションのスケーリング[kWh]（-1.0〜1.0 を -inverter_max_cap〜inverter_max_cap に変換）
            # 既存のRLモデルの行動空間は[-1, 1]であるため、ここでスケーリングする
            # _net_action = action[0] * self.env.inverter_max_cap * 0.5  # [kWh]
            # 行動空間をインバーターに合わせた[-2,2]に変更したときは上記コードは消す
            # _next_soc = current_soc - (_net_action/self.env.battery_max_cap) # [割合]
            # SoCをリストに追加
            self.soc_list.append(next_soc)
            # アクションをリストに追加
            self.natural_action_list.append(action[0]) # 純粋なRLモデルの行動
            self.edited_action_list.append(edited_action) # 編集後の行動
            # 電力取引量をリストに追加
            self.energytransfer_list.append(energytransfer)

        # 環境のクローズ
        self.env_bid.close()

        # テスト結果のデータフレームを作成
        # 元のテストデータを使用し、アクションとSoCを追加
        df_testresult = self.df_test_original
        df_testresult = df_testresult.iloc[:len(self.edited_action_list)].reset_index(drop=True)

        #--------------------for debug----------------------------#
        df_soc = pd.DataFrame(self.soc_list, columns=[f"SoC_{mode}(include initial SoC)"])
        df_action = pd.DataFrame(self.action_debug_list, columns=[f"action_{mode}_fromRL"])
        df_combined = pd.concat([df_soc, df_action], axis=1)
        df_combined.to_csv("Battery-Control-By-Reinforcement-Learning/for_debug/debug.csv", header=True, index=False)
        #--------------------------------------------------------#

        # socを[%]に変換して追加して、初期SoCはリストの最初にあるため、1つ目を除く
        df_testresult[f"SoC_{mode}[%]"] = [x * 100 for x in self.soc_list[1:]]
        df_testresult[f"charge/discharge_{mode}[kWh]"] = self.edited_action_list
        df_testresult[f"natural_action_{mode}[kWh]"] = self.natural_action_list
        df_testresult[f"energytransfer_{mode}[kWh]"] = self.energytransfer_list

        return df_testresult
    
    # リアルタイムテスト
    def dispatch_test_realtime(self, model_name, mode):
        # 学習済みモデルのロード
        self.model = PPO.load(model_name)

        # テストデータの各ステップについてループ
        for idx_state in range(len(self.df_test)):
            # 現在のタイムステップを取得
            time_of_day = idx_state % self.env_realtime.day_steps
            theta = 2 * math.pi * time_of_day / self.env_realtime.day_steps
            sin_time = math.sin(theta)
            cos_time = math.cos(theta)

            # 現在のSoCを取得
            current_soc = self.soc_list[-1] # 0.0 ~ 1.0[割合]
            PV_predict = self.df_test.at[idx_state, f"PV_predict_{mode}[kW]"]

            # 観測値の構築
            obs = [
                self.df_test.at[idx_state, f"PV_predict_{mode}_normalized"], # [正規化]
                self.df_test.at[idx_state, f"energyprice_predict_{mode}_normalized"], # [正規化]
                self.df_test.at[idx_state, f"imbalanceprice_predict_{mode}_normalized"], # [正規化]
                self.df_test.at[idx_state, f"energytransfer_bid_normalized"],  # 前日の入札値[正規化]
                current_soc,  # [正規化]
                sin_time, # [正規化]
                cos_time # [正規化]
            ]

            # モデルによるアクションの予測
            action, _states = self.model.predict(obs, deterministic=True)

            # for debug
            self.action_debug_list.append(action)

            edited_action, next_soc, energytransfer = Operate().operate_plan(PV_predict, action[0], current_soc)
            # アクションのスケーリング[kWh]（-1.0〜1.0 を -inverter_max_cap〜inverter_max_cap に変換）
            # 既存のRLモデルの行動空間は[-1, 1]であるため、ここでスケーリングする
            # _net_action = action[0] * self.env.inverter_max_cap * 0.5  # [kWh]
            # 行動空間をインバーターに合わせた[-2,2]に変更したときは上記コードは消す
            # _next_soc = current_soc - (_net_action/self.env.battery_max_cap) # [割合]
            # SoCをリストに追加
            self.soc_list.append(next_soc)
            # アクションをリストに追加
            self.natural_action_list.append(action[0]) # 純粋なRLモデルの行動
            self.edited_action_list.append(edited_action) # 編集後の行動
            # 電力取引量をリストに追加
            self.energytransfer_list.append(energytransfer)

        # 環境のクローズ
        self.env_realtime.close()

        # テスト結果のデータフレームを作成
        # 元のテストデータを使用し、アクションとSoCを追加
        df_testresult = self.df_test_original
        df_testresult = df_testresult.iloc[:len(self.edited_action_list)].reset_index(drop=True)

        #--------------------for debug----------------------------#
        df_soc = pd.DataFrame(self.soc_list, columns=[f"SoC_{mode}(include initial SoC)"])
        df_action = pd.DataFrame(self.action_debug_list, columns=[f"action_{mode}_fromRL"])
        df_combined = pd.concat([df_soc, df_action], axis=1)
        df_combined.to_csv("Battery-Control-By-Reinforcement-Learning/for_debug/debug.csv", header=True, index=False)
        #--------------------------------------------------------#

        # socを[%]に変換して追加して、初期SoCはリストの最初にあるため、1つ目を除く
        df_testresult[f"SoC_{mode}[%]"] = [x * 100 for x in self.soc_list[1:]]
        df_testresult[f"charge/discharge_{mode}[kWh]"] = self.edited_action_list
        df_testresult[f"natural_action_{mode}[kWh]"] = self.natural_action_list
        df_testresult[f"energytransfer_{mode}[kWh]"] = self.energytransfer_list
        return df_testresult
    
    def dispatch_testresult_update(self, df_test):
        year, month, day = self.get_current_date()
        print(f'year: {year}, month: {month}, day: {day}')
        # 過去予測・実績データを読み込む
        df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        # 現在の日付(別モード）がすでに存在するか確認（存在する ＝ True, 存在しない = False）
        predictdate_exists = ((df_original['year'] == year) &
                              (df_original['month'] == month) &
                              (df_original['day'] == day)).any()
        # year, month, day, hourのをindexとして設定
        if predictdate_exists: # すでに存在する場合
            # year, month, day, hourをindexに設定
            df_original.set_index(['year', 'month', 'day', 'hour'], inplace = True)
            df_test.set_index(['year', 'month', 'day', 'hour'], inplace = True)
            # 同じ日付データグループを更新
            df_original.update(df_test)
            # indexを振りなおす
            df_original.reset_index(inplace = True)
            df_test.reset_index(inplace = True)
            df_original.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", header = True, index = False)
        else: # 存在しない場合
            # df_testと同じ形のデータセットを用意
            df_result = pd.DataFrame(index = range(len(df_test)), columns = df_original.columns)
            # 重なるcolomunだけ更新
            df_result.update(df_test.reset_index(drop = True))
            # result_dataframe.csvの下から追加
            df_result.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", mode = 'a', header = False, index = False)
    
    def mode_dependent_test(self, latestModel_name, mode):
        df_test = getattr(self, f"dispatch_test_{mode}")(latestModel_name, mode)
        print(f"{mode}_test")
        self.dispatch_testresult_update(df_test)