# 外部モジュール
import warnings
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# 内製モジュール
from RL_env import ESS_ModelEnv as Env
from RL_dataframe_manager import Dataframe_Manager as Dfmanager
# Errorメッセージを非表示にする
warnings.simplefilter('ignore')

# 学習済みの強化学習モデルのテスト（評価）を行うクラス
class TestModel():
    def __init__(self, mode):
        # クラスのインスタンス化
        self.env = Env(mode) # 
        self.dfmanager = Dfmanager() # データ読込みクラスのインスタンス化
        # soc_list,action_listを初期化
        self.soc_list = [0.5] # SoCの初期値
        self.action_list = []

    # 学習したモデルをテスト
    def dispatch_bid_test(self, model_name):
        # 学習したモデルをロード
        self.model = PPO.load(model_name)
        # testデータの読込み
        df_test = self.dfmanager.get_test_df_bid()
        # observationの項目を定義
        col = ["PV_predict_bid[kW]", "energyprice_predict_bid[Yen/kWh]", "imbalanceprice_predict_bid[Yen/kWh]", "SoC_bid[%]", "charge/discharge_bid[kWh]"]
        # df_testのcolの列の最初の行をリストに変換してobs_listに格納
        obs_list = [df_test[col].iloc[0].tolist()]
        # testデータのコマ数分(len(df_test))だけ、学習済みのモデルによるactionを得る
        for idx_state in range(len(df_test)):
            # df_test内のPV_predict_bid, energyprice_predict_bid, imbalanceprice_predict_bidの48コマ分のデータを抽出
            # - 取得する行数はtestデータのコマ数分
            # - SoCは最新のものを読み込む（すでに１日立っていれば、前日の最終SoCを使うことになる）
            obs = [
                df_test["PV_predict_bid[kW]"][idx_state].astype(float),
                df_test["energyprice_predict_bid[Yen/kWh]"][idx_state].astype(float),
                df_test["imbalanceprice_predict_bid[Yen/kWh]"][idx_state].astype(float),
                self.soc_list[-1] # SoCの最新のもの[-1]を使う
            ]
            # actionを得るためには、学習済みのLSTMモデルへobservationを入れるだけ。rewardが必要無いので、step関数は使わない
            # actionを得るためには、学習済みのLSTMモデルへtestデータのobservationを入れるだけ。rewardが必要無いので、step関数は使わない
            obs_array = np.array(obs)
            action, _ = self.model.predict(obs_array)    
            # actionによって得られる次のコマのobservationとrewardを計算する。
            # step関数を使う。ただし、stepはtest用のデータを使うのでmethodが別で用意されている。
            # SoCを更新してリストへ追加
            self.soc_list.append(self.soc_list[-1] + action[0])
            # obs,actionをそれぞれリストに追加
            obs_list.append(obs)
            self.action_list.append(action)

        # 環境のクローズ
        self.env.close()
        # test結果を返す
        # obs_listとself.action_listを結合して、df_testresultを作成
        # obs_listは、self.action_listより1つ多いので、最初の要素を削除する
        obs_list.pop(0)
        df_testresult = pd.DataFrame(obs_list, columns=["PV_predict_bid[kW]", "energyprice_predict_bid[Yen/kWh]", "imbalanceprice_predict_bid[Yen/kWh]", "SoC_bid[%]"])
        df_testresult["charge/discharge_bid[kWh]"] = pd.DataFrame(self.action_list)
        df_test.update(df_testresult)
        return df_test, df_testresult
    
    # 学習したモデルをテスト
    def dispatch_realtime_test(self, model_name):
        # 学習したモデルをロード
        self.model=PPO.load(model_name)
        # testデータの読込み
        df_test = self.dfmanager.get_test_df_realtime()
        # observationの項目を定義
        col = ["PV_predict_realtime[kW]", "energyprice_predict_realtime[Yen/kWh]", "imbalanceprice_predict_realtime[Yen/kWh]", "SoC_realtime[%]", "charge/discharge_realtime[kWh]"]
        # df_testのcolの列の最初の行をリストに変換してobs_listに格納
        obs_list = [df_test[col].iloc[0].tolist()]
        # testデータのコマ数分(len(df_test))だけ、学習済みのモデルによるactionを得る
        for idx_state in range(len(df_test)):
            # df_test内のPV_predict_bid, energyprice_predict_bid, imbalanceprice_predict_bidの48コマ分のデータを抽出
            # - 取得する行数はtestデータのコマ数分
            # - SoCは最新のものを読み込む（すでに１日立っていれば、前日の最終SoCを使うことになる）
            obs = [
                df_test["PV_predict_realtime[kW]"][idx_state].astype(float),
                df_test["energyprice_predict_realtime[Yen/kWh]"][idx_state].astype(float),
                df_test["imbalanceprice_predict_realtime[Yen/kWh]"][idx_state].astype(float),
                self.soc_list[-1]
            ]
            # actionを得るためには、学習済みのLSTMモデルへobservationを入れるだけ。rewardが必要無いので、step関数は使わない
            # actionを得るためには、学習済みのLSTMモデルへtestデータのobservationを入れるだけ。rewardが必要無いので、step関数は使わない
            obs_array = np.array(obs)
            action, _ = self.model.predict(obs_array)
             # actionによって得られる次のコマのobservationとrewardを計算する。
            # step関数を使う。ただし、stepはtest用のデータを使うのでmethodが別で用意されている。
            # SoCを更新してリストへ追加
            self.soc_list.append(self.soc_list[-1] + action[0])
            obs_list.append(obs)
            self.action_list.append(action)

        # 環境のクローズ
        self.env.close()
        # test結果を返す
        # obs_listとself.action_listを結合して、df_testresultを作成
        # obs_listは、self.action_listより1つ多いので、最初の要素を削除する
        obs_list.pop(0)
        df_testresult = pd.DataFrame(obs_list, columns=["PV_predict_realtime[kW]", "energyprice_predict_realtime[Yen/kWh]", "imbalanceprice_predict_realtime[Yen/kWh]", "SoC_realtime[%]"])
        df_testresult["charge/discharge_realtime[kWh]"] = pd.DataFrame(self.action_list)
        df_test.update(df_testresult)
        return df_test, df_testresult
    
    def dispatch_testresult_update(self, df_test, df_testresult):
        date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        # date_infoは {'year': year, 'month': month, 'day': day} の形式
        date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
        latest_date = date_info['date'].max()

        year = latest_date.year
        month = latest_date.month
        day = latest_date.day

        # RLの結果を書き込んだファイル(result_dataframe.csv)を読み込む
        df_original_result = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        # 最新日付のデータをフィルタリング(indexを0から再配置)
        df_result = df_original_result[(df_original_result['year'] == year) & 
                                     (df_original_result['month'] == month) & 
                                      (df_original_result['day'] == day)].reset_index(drop=True)
        # df_resultの列を保持しつつ、行数をdf_testresultに一致させる
        df_result = pd.DataFrame(index = range(len(df_testresult)), columns = df_original_result.columns)
        df_result.update(df_test)
        return df_result, df_testresult
    
    def mode_dependent_test(self, latestModel_name, mode):
        if mode == "bid":
            df_test, df_testresult = self.dispatch_bid_test(latestModel_name)
            print("bid_test")
        elif mode == "realtime":
            df_test, df_testresult = self.dispatch_realtime_test(latestModel_name)
            print("realtime_test")

        df_result, df_testresult = self.dispatch_testresult_update(df_test, df_testresult)
        print(df_testresult)
        print(df_test)
        print("test結果: ", df_result)
        df_result.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", mode='a', header=False, index=False)
        return df_testresult #なぜかdf_testresultを返している、いらないかもしれない
