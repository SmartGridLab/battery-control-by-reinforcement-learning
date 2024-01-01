# 外部モジュール
import warnings
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
#from torch.utils.tensorboard import SummaryWriter # tensorBoardを起動して、学習状況を確認する

# 内製モジュール
import RL_env as env

# Errorメッセージを非表示にする
warnings.simplefilter('ignore')

class TestModel:
    # モデルのテスト
    def dispatch_test(self, model_name):
        # 学習したモデルをロード
        self.model = PPO.load(model_name)
        # 学習したモデルをテスト
        obs = env.reset() # stateをリセットする
        obs = pd.Series(obs)
        obs = torch.tensor(obs.values.astype(np.float64))
        for i in range(0, self.test_days - 1):
            action, _ = self.model.predict(obs)
            obs, reward, done, _ = self.step(action)
            obs = pd.Series(obs)
            obs = torch.tensor(obs.values.astype(np.float64))