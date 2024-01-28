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
        col = ["PV_predict_bid", "energyprice_predict_bid", "imbalanceprice_predict_bid", "SoC_bid", "charge/discharge_bid"]
        # df_testのcolの列の最初の行をリストに変換してobs_listに格納
        self.obs_list = [self.df_test[col].iloc[0].tolist()]
        print("initial obs_list: ", self.obs_list)

    # 学習したモデルをテスト
    def dispatch_test(self, model_name):      
        # 学習したモデルをロード
        self.model = PPO.load(model_name)

        # testデータのコマ数分(len(self.df_test))だけ、学習済みのモデルによるactionを得る
        for idx_state in range(len(self.df_test)):
            # obsだけをスライスして出力
            print("idx_state: ", idx_state+1, "/", len(self.df_test))
            # print("df_test: ", self.df_test.iloc[idx_state])
            ## self.obs_listとsocを結合して、obs_listを更新
            #self.obs_list[idx_state].append(self.soc_list[-1])
            #print("obs_list[index_state]: ", self.obs_list[idx_state])

            # df_test内のPV_predict_bid, energyprice_predict_bid, imbalanceprice_predict_bidの48コマ分のデータを抽出
            # - 取得する行数はstate_idx(当該time_step)から48コマ分
            # - SoCは最新のものを読み込む（すでに１日立っていれば、前日の最終SoCを使うことになる）
            # 要素が47個ですべて-1であるリストを作成, SoCの最新版だけくっつける
            # NaNlist = [-1 for i in range(48)]
            # print("PV_predict_bid: ", self.df_test["PV_predict_bid"][idx_state].astype(float),)
            obs = [
                self.df_test["PV_predict_bid"][idx_state].astype(float),
                self.df_test["energyprice_predict_bid"][idx_state].astype(float),
                self.df_test["imbalanceprice_predict_bid"][idx_state].astype(float),
                self.soc_list[-1] # SoC
            ]
            # actionを得るためには、学習済みのLSTMモデルへobservationを入れるだけ。rewardが必要無いので、step関数は使わない
            action, _ = self.model.predict(obs)    
            obs, reward, done, _ = self.env.step(float(action))    # このstepの戻り値のrewardは使えない（_get_reward内で参照しているデータが数値がtrainingのものになってしまっているはずなので）
            # obsをself.obs_listに追加
            self.obs_list.append(obs)
            self.action_list.append(action)
            # print("obs_list: ", self.obs_list)
        # 環境のクローズ
        self.env.close()

        # test結果を返す
        # self.obs_listとself.action_listを結合して、df_testresultを作成
        # - self.obs_listは、self.action_listより1つ多いので、最初の要素を削除する
        self.obs_list.pop(0)
        df_testresult = pd.DataFrame(self.obs_list, columns=["PV_predict_bid", "energyprice_predict_bid", "imbalanceprice_predict_bid", "SoC_bid"])
        df_testresult["charge/discharge_bid"] = pd.DataFrame(self.action_list)        
        
        return df_testresult