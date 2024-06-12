# 外部モジュール
import warnings
import pandas as pd
from stable_baselines3 import PPO

# 内製モジュール
from RL_env import ESS_ModelEnv as Env
from RL_dataframe_manager import Dataframe_Manager as Dfmanager
# Errorメッセージを非表示にする
warnings.simplefilter('ignore')

# 学習済みの強化学習モデルのテスト（評価）を行うクラス
class TestModel:
    def __init__(self):
        # クラスのインスタンス化
        self.env = Env() # 
        self.dfmanager = Dfmanager() # データ読込みクラスのインスタンス化
        # testデータの読込み
        self.df_test = self.dfmanager.get_test_df()
        # soc_list,action_listを初期化
        self.soc_list = [0.5] # SoCの初期値
        self.action_list = []
        # observationの項目を定義
        col = ["PV_predict_bid[kW]", "energyprice_predict_bid[Yen/kWh]", "imbalanceprice_predict_bid[Yen/kWh]", "SoC_bid[%]", "charge/discharge_bid[kWh]"]
        # df_testのcolの列の最初の行をリストに変換してobs_listに格納
        self.obs_list = [self.df_test[col].iloc[0].tolist()]
        # print("initial obs_list: ", self.obs_list)

    # 学習したモデルをテスト
    def dispatch_test(self, model_name):      
        # 学習したモデルをロード
        self.model = PPO.load(model_name)

        # testデータのコマ数分(len(self.df_test))だけ、学習済みのモデルによるactionを得る
        for idx_state in range(len(self.df_test)):
            # df_test内のPV_predict_bid, energyprice_predict_bid, imbalanceprice_predict_bidの48コマ分のデータを抽出
            # - 取得する行数はtestデータのコマ数分
            # - SoCは最新のものを読み込む（すでに１日立っていれば、前日の最終SoCを使うことになる）
            obs = [
                self.df_test["PV_predict_bid[kW]"][idx_state].astype(float),
                self.df_test["energyprice_predict_bid[Yen/kWh]"][idx_state].astype(float),
                self.df_test["imbalanceprice_predict_bid[Yen/kWh]"][idx_state].astype(float),
                self.soc_list[-1] # SoC
            ]
            # actionを得るためには、学習済みのLSTMモデルへobservationを入れるだけ。rewardが必要無いので、step関数は使わない

            action, _ = self.model.predict(obs)    
            obs, reward, done, _ = self.env.step(float(action))    # このstepの戻り値のrewardは使えない（_get_reward内で参照しているデータが数値がtrainingのものになってしまっているはずなので）
            # obsをself.obs_listに追加
            self.obs_list.append(obs)
            self.action_list.append(action)
        # 環境のクローズ
        self.env.close()

        # test結果を返す
        # self.obs_listとself.action_listを結合して、df_testresultを作成
        # - self.obs_listは、self.action_listより1つ多いので、最初の要素を削除する
        self.obs_list.pop(0)
        df_testresult = pd.DataFrame(self.obs_list, columns=["PV_predict_bid[kW]", "energyprice_predict_bid[Yen/kWh]", "imbalanceprice_predict_bid[Yen/kWh]", "SoC_bid[%]"])
        df_testresult["charge/discharge_bid[kWh]"] = pd.DataFrame(self.action_list)        
        
        return df_testresult