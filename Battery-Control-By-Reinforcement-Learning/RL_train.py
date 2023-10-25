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
print("\n---充放電計画策定プログラム開始---\n")

class TrainModel:
    def __init__(self,env):
        self.env = env
        self.path = path # 保存先のpathを指定


    # モデルのトレーニング
    def dispatch_train(self):
        # [2023.09.07 小平の理解は以下の通り]
        # action_space: 何コマ先までの充放電を考慮するか
        # steps: 最適運用スケジュールを行う期間に対応（48だと48コマ分の充放電スケジュールを作成する）
        # episode: steps分のコマ数を１episodeとして、何回試すか（ランダム要素を評価するため）
        # num_act_window: action_space(12コマ)num_act_window(4batch)行って、1日48コマに相当する（１日分で1episode)
        print("-モデル学習開始-")
        model = PPO("MlpPolicy", self.env, gamma = 0.8, gae_lambda = 1, clip_range = 0.2, 
                        ent_coef = 0.005, vf_coef =0.5, learning_rate = 0.0001, n_steps = 48, 
                        verbose=0, tensorboard_log="./PPO_tensorboard/") 
        model.learn(total_timesteps=20000)
        print("-モデル学習終了-")

        # 学習済みモデルの保存
        model.save(self.path)

        # じゃんけん環境のクローズ
        env.close()

    # 入力データの設定
    def data_set(self):
        self.PV_out_time = self.PVout[self.time]
        self.price_time = self.price[self.time]
        self.imbalance_time = self.imbalance[self.time]
        
        #過去の実測値から最大値を取得し格納
        if self.days != self.last_day:
            self.MAX_price = max(self.price[48*(self.days - 1):48*self.days])
        
        #時刻self.timeに対応するデータを取得
        self.input_price = self.price[48*(self.days - 1) + self.time]
        self.input_PV = self.PVout[48*(self.days - 1) + self.time]
        
        #self.timeを起点にしてアクションスペース分のデータを取得
        self.input_PV_data = (self.PVout[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/2).T[0]
        self.input_price_data = (self.price[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/self.MAX_price).T[0]
        self.input_imbalance_data = (self.imbalance[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/self.MAX_price).T[0]

