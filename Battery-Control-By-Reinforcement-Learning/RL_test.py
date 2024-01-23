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
        self.soc_init = 0.5 # SoCの初期値
        # soc_list,action_listを初期化
        self.soc_list = []
        self.action_list = []

    # 学習したモデルをテスト
    def dispatch_test(self, model_name):      
        # 学習したモデルをロード
        self.model = PPO.load(model_name)

        # testデータのコマ数分だけ、学習済みのモデルによるactionを得る
        for idx_state in range(len(self.df_test)):
            # obsだけをスライスして出力
            print("df_test: ", self.df_test.iloc[idx_state])
            # self.df_test.iloc[0]をpandas形式からfloatのリストに変換
            self.obs_init = self.df_test.iloc[idx_state].tolist()
            # self.df_testとsoc_initを結合して、self.df_testに格納
            self.obs_init.append(self.soc_init)
            print("obs_init: ", self.obs_init)

            # df_test内のPV_predict_bid, energyprice_predict_bid, imbalanceprice_predict_bidの48コマ分のデータを抽出
            # - 取得する行数はstate_idx(当該time_step)から48コマ分
            # - SoCは最新のものを読み込む（すでに１日立っていれば、前日の最終SoCを使うことになる）
            # 要素が47個ですべて-1であるリストを作成, SoCの最新版だけくっつける
            # NaNlist = [-1 for i in range(48)]
            obs_init = [
                self.df_test["PV_predict_bid"][i],
                self.df_test["energyprice_predict_bid"][i],
                self.df_test["imbalanceprice_predict_bid"][i],
                self.soc_list[-1] # SoC
            ]




            # actionを得るためには、学習済みのLSTMモデルへobservationを入れるだけ。rewardが必要無いので、step関数は使わない
            action, _ = self.model.predict(obs["PV_predict_bid"][i],
                                           obs["energyprice_predict_bid"][i],
                                           obs["imbalanceprice_predict_bid"][i],
                                           obs["SoC_bid"][i])    
            obs, reward, done, _ = self.env.step(float(action))    # このstepの戻り値のrewardは使えない（_get_reward内で参照しているデータが数値がtrainingのものになってしまっているはずなので）
            # socをリストに格納
            print('latest SoC:', obs[-1][-1])
            self.soc_list.append(obs[-1][-1])
            # actionをリストに格納(charge/discharge)
            self.action_list.append(float(action))    
            print("action: ", self.action_list)
            print("obs: ", self.soc_list)
        # 環境のクローズ
        self.env.close()
        # test結果を返す
        # soc_list, action_listをpandasのDataFrame形式に変換して結合
        # df_testresultのheaderはSoC_bid, charge/discharge_bidとする
        soc_list = pd.DataFrame(self.soc_list)
        action_list = pd.DataFrame(self.action_list)
        df_testresult = pd.concat([soc_list, action_list], axis=1)
        # df_testresultのheaderはSoC_bid, charge/discharge_bidとする
        df_testresult.columns = ["PV_predict_bid", "energyprice_predict_bid", "imbalanceprice_predict_bid", "SoC_bid", "charge/discharge_bid"]
        return df_testresult