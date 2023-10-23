# 外部モジュール
import gym
import warnings
import numpy as np
import pandas as pd
import torch
import math as ma
import tkinter as tk
from stable_baselines3 import PPO
#from torch.utils.tensorboard import SummaryWriter # tensorBoardを起動して、学習状況を確認する

# 内製モジュール
import RL_env as env

warnings.simplefilter('ignore')


class TestModel:
    # モデルのテスト
    def test_trainedModel(self, mode, train_days, episode, model_name):
        # 学習したモデルをロード
        self.model = PPO.load(model_name)
        # 学習したモデルをテスト
        obs = env.reset() # stateをリセットする
        obs = pd.Series(obs)
        obs = torch.tensor(obs.values.astype(np.float64))
        for i in range(0, self.test_days - 1)):
            action, _ = self.model.predict(obs)
            obs, reward, done, _ = self.step(action)
            obs = pd.Series(obs)
            obs = torch.tensor(obs.values.astype(np.float64))

    # 入力データの設定
    def data_set(self):
        self.PV_out_time = self.PVout[self.time]
        self.price_time = self.price[self.time]
        self.imbalance_time = self.imbalance[self.time]
        
            #過去の実測値から最大値を取得し格納
            if self.days != self.last_day:
                self.MAX_price = max(self.price)

            #時刻self.timeに対応するデータを取得
            self.input_PV = self.PVout[self.time]
            self.input_price = self.price[self.time]
            
            #self.timeを起点にしてアクションスペース分のデータを取得
            self.input_PV_data = (self.PVout[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/2).T[0]
            self.input_price_data = (self.price[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/self.MAX_price).T[0]
            self.input_imbalance_data = (self.imbalance[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/self.MAX_price).T[0]
