# 外部モジュール
import warnings
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
#from torch.utils.tensorboard import SummaryWriter # tensorBoardを起動して、学習状況を確認する

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
        # obs_list,action_listを初期化
        self.obs_list = []
        self.action_list = []

        # 学習したモデルをテスト
        obs = self.env.reset_forTest() # stateをリセットする
        obs = pd.Series(obs)
        obs = torch.tensor(obs.values.astype(np.float64))
        # obsの行数分だけforを回す
        for i in range(len(obs)):
            # actionを得るためには、学習済みのLSTMモデルへobservationを入れるだけ。rewardが必要無いので、step関数は使わない
            # obsがSoCだけなので、モデルはSoCのみでactionを決定することになる(PVや電力価格予測をobsとしては使っていないことになる・・・)
            # ただし、modelの学習時にはPVや電力価格予測をrewardの計算に使っているので、PVや電力価格予測を使ってactionを決定していることになる？）
            # 本質的にはobsにPVの発電量予測値や電力価格予測値を入れるべきと思われる（要検討）
            action, _ = self.model.predict(obs)                
            obs, reward, done, _ = self.env.step(action)    # このstepの戻り値のrewardは使えない（_get_reward内で参照しているデータが数値がtrainingのものになってしまっているはずなので）
            obs = pd.Series(obs)
            obs = torch.tensor(obs.values.astype(np.float64))    
            # obsをリストに格納
            self.obs_list.append(obs)
            # actionをリストに格納
            self.action_list.append(action)    
        # 環境のクローズ
        self.env.close()
        # test結果を返す
        # obs_list: SoCの値が入ったリスト
        # action_list: charge\dischargeの値が入ったリスト
        return self.obs_list, self.action_list