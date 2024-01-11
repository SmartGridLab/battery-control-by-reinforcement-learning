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
        
        # 学習したモデルをテスト
        obs = self.env.reset_forTest() # stateをリセットする
        obs = pd.Series(obs)
        obs = torch.tensor(obs.values.astype(np.float64))
        # obsの行数分だけforを回す
        for i in range(len(obs)):
            action, _ = self.model.predict(obs) # actionを得るためには、学習済みのLSTMモデルへobservationを入れるだけ。rewardが必要無いので、step関数は使わない
            obs, reward, done, _ = self.env.step(action)    # このstepの戻り値のrewardは使えない（_get_reward内で参照しているデータが数値がtrainingのものになってしまっているはずなので）
            obs = pd.Series(obs)
            obs = torch.tensor(obs.values.astype(np.float64))    
        # 環境のクローズ
        self.env.close()
