# 外部モジュール
import warnings
import pandas as pd
import math
from stable_baselines3 import PPO
import numpy as np

# 内製モジュール
from RL_env import ESS_ModelEnv as Env
from RL_dataframe_manager import Dataframe_Manager as Dfmanager

warnings.simplefilter('ignore')

# 学習済みの強化学習モデルのテスト（評価）を行うクラス
class TestModel:
    def __init__(self):
        # 環境のインスタンス化
        self.env = Env()
        self.dfmanager = Dfmanager()  # データ読込みクラスのインスタンス化
        
        # 学習用のデータ、テストデータ、結果格納テーブルを取得
        self.df_train = self.env.df_train
        self.df_test_original = self.dfmanager.get_test_df()  # 正規化前の元データを保持
        self.df_test = self.df_test_original.copy()

        # デバッグ用：データフレームのカラム名を出力
        print("Test DataFrame Columns:", self.df_test.columns.tolist())
        
        # 正規化パラメータの取得
        self.pvout_max = self.env.pvout_max
        self.pvout_min = self.env.pvout_min
        self.price_max = self.env.price_max
        self.price_min = self.env.price_min
        self.imbalance_max = self.env.imbalance_max
        self.imbalance_min = self.env.imbalance_min

        # テストデータの正規化
        self.df_test['PV_predict_bid[kW]'] = (self.df_test['PV_predict_bid[kW]'] - self.pvout_min) / (self.pvout_max - self.pvout_min)
        self.df_test['energyprice_predict_bid[Yen/kWh]'] = (self.df_test['energyprice_predict_bid[Yen/kWh]'] - self.price_min) / (self.price_max - self.price_min)
        self.df_test['imbalanceprice_predict_bid[Yen/kWh]'] = (self.df_test['imbalanceprice_predict_bid[Yen/kWh]'] - self.imbalance_min) / (self.imbalance_max - self.imbalance_min)

        # 正規化後のテストデータの確認（オプション）
        print("self.df_test (normalized): ", self.df_test.head())

        # SoCとアクションのリストを初期化
        self.soc_list = [0.5]  # SoCの初期値（50%）
        self.action_list = []
        
    # 学習したモデルをテスト
    def dispatch_test(self, model_name):      
        # 学習済みモデルのロード
        self.model = PPO.load(model_name)

        # テストデータの各ステップについてループ
        for idx_state in range(len(self.df_test)):
            # 現在のタイムステップを取得
            time_of_day = idx_state % self.env.day_steps
            theta = 2 * math.pi * time_of_day / self.env.day_steps
            sin_time = math.sin(theta)
            cos_time = math.cos(theta)

            # 現在のSoCを取得
            current_soc = self.soc_list[-1]

            # 観測値の構築
            obs = [
                self.df_test["PV_predict_bid[kW]"][idx_state],
                self.df_test["energyprice_predict_bid[Yen/kWh]"][idx_state],
                self.df_test["imbalanceprice_predict_bid[Yen/kWh]"][idx_state],
                current_soc,  # SoC
                sin_time,
                cos_time
            ]

            # モデルによるアクションの予測
            action, _states = self.model.predict(obs, deterministic=True)

            # アクションのスケーリング[kWh]（-1.0〜1.0 を -inverter_max_cap〜inverter_max_cap に変換）
            action_scaled = action[0] * self.env.inverter_max_cap * 0.5  # スケーリングファクターは環境に合わせる

            # SoCの更新
            new_soc = current_soc - (action_scaled / self.env.battery_max_cap)
            # SoCを0から1の範囲にクリップ
            new_soc = max(0.0, min(new_soc, 1.0))
            # SoCをリストに追加
            self.soc_list.append(new_soc)

            # アクションをリストに追加
            self.action_list.append(action_scaled)

        # 環境のクローズ
        self.env.close()

        # テスト結果のデータフレームを作成
        # 元のテストデータを使用し、アクションとSoCを追加
        df_testresult = self.df_test_original
        df_testresult = df_testresult.iloc[:len(self.action_list)].reset_index(drop=True)
        # socを[%]に変換して追加して、初期SoCはリストの最初にあるため、1つ目を除く
        df_testresult["SoC_bid[%]"] = [x * 100 for x in self.soc_list[1:]]
        df_testresult["charge/discharge_bid[kWh]"] = self.action_list

        print("df_testresult: ", df_testresult)

        # 既存の結果ファイルを読み込み
        df_original_result = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        # df_testresult に存在しないカラムを NaN で埋める
        for col in df_original_result.columns:
            if col not in df_testresult.columns:
                df_testresult[col] = np.nan  # または適切なデフォルト値

        # df_testresult を df_original_result のカラム順に並べ替える
        df_testresult = df_testresult[df_original_result.columns]

        # 結果の確認（オプション）
        print("test結果: ", df_testresult.head())

        # 結果をCSVファイルに保存（既存ファイルに追記）
        df_testresult.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", mode='a', header=False, index=False)
        
        return df_testresult
