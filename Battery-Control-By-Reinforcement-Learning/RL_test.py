# 外部モジュール
import warnings
import pandas as pd
from stable_baselines3 import PPO

# 内製モジュール
from RL_env import ESS_ModelEnv as env

# Errorメッセージを非表示にする
warnings.simplefilter('ignore')

# 学習済みの強化学習モデルのテスト（評価）を行うクラス
class TestModel:
    def __init__(self):
        self.env = env() # 環境のインスタンス化

    def dispatch_test(self, model_name):
        # 学習したモデルをロード
        self.model = PPO.load(model_name)
        # soc_list,action_listを初期化
        self.soc_list = []
        self.action_list = []

        # 学習したモデルをテスト
        obs = self.env.reset_forTest() # stateをリセットする
        # testデータのコマ数分だけ、学習済みのモデルによるactionを得る
        for i in range(len(obs)):
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