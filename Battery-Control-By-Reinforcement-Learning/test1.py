import gym
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math as ma
#import tensorflow as tf

from matplotlib.backends.backend_pdf import PdfPages
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter  # tensorBoardを起動して、学習状況を確認する

warnings.simplefilter('ignore')

class ESS_Model(gym.Env):
    def __init__(self, mode, pdf_day, train_days, test_day, PV_parameter, action_space):
        # パラメータの定義
        self.episode = 0
        self.total_step = action_space  # 1Dayの総コマ数
        self.gamma = ma.exp(-(1 / action_space))  # 放電に対する割引率
        self.omega = ma.exp(1 / action_space)  # 充電に対する割引率
        self.battery_MAX = 4  # ４kWh
        self.MAX_reward = -10000
        self.K = 1.46  # インバランス料金算出のパラメータ
        self.L = 0.43  # インバランス料金算出のパラメータ
        self.Train_Days = train_days  # 学習Day
        self.test_days = test_day - 1  # テストDay数
        self.mode = mode
        if mode == "train":
            self.last_day = self.Train_Days
        elif mode == "test":
            self.last_day = self.test_days
        self.all_rewards = []
        # データのロード
        input_data = pd.read_csv("input_data2022.csv")
        predict_data = pd.read_csv("price_predict.csv")

        # 学習(テスト)用データ作成
        if self.mode == "train":
            input_data = input_data[:self.Train_Days * 48]
        elif self.mode == "test":
            input_data = input_data[self.Train_Days * 48:(self.Train_Days + self.test_days + 1) * 48]

        input_data = input_data[["year", "month", "day", "hour", "hourSin", "hourCos", "upper", "lower", "PVout", "price", "imbalance"]]
        predict_data = predict_data[["year", "month", "day", "hour", "hourSin", "hourCos", "upper", "lower", "PVout", "price", "imbalance"]]

        self.input_data = input_data.values
        self.predict_data = predict_data.values

        # アクション
        self.ACTION_NUM = action_space  # アクションの数(現状は48の約数のみ)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.ACTION_NUM,))

        # 状態の上限と下限の設定
        low_box = np.zeros(self.ACTION_NUM*2+1) # 入力データの下限値×入力データの数
        high_box = np.ones(self.ACTION_NUM*2+1) # 入力データの上限値×入力データの数
        LOW = np.array(low_box)
        HIGH = np.array(high_box)
        self.observation_space  = gym.spaces.Box(low=LOW, high=HIGH)
        # 初期データの設定
        self.reset()

        # パラメータの設定
        self.day = 0
        self.step = 0
        self.battery = 0  # バッテリーの初期値
        self.PV_max = PV_parameter[0]  # PV最大発電量
        self.PV_min = PV_parameter[1]  # PV最小発電量
        self.PV = 0  # PV発電量
        self.price_predict = 0  # 価格予測値
        self.price_real = 0  # 価格実績値
        self.imbalance = 0  # インバランス料金
        self.state = np.zeros(self.ACTION_NUM * 2 + 1)  # 状態変数
        self.reward = 0  # 報酬
        self.done = False  # エピソード完了フラグ
        self.info = {}  # 追加情報

        self.writer = SummaryWriter()  # ログを出力するための設定

    def reset(self):
        self.day = 0
        self.step = 0
        self.battery = 0  # バッテリーの初期値
        self.PV = self.input_data[self.day * self.total_step + self.step][8]  # PV発電量の取得
        self.price_predict = self.predict_data[self.day * self.total_step + self.step][9]  # 価格予測値の取得
        self.price_real = self.input_data[self.day * self.total_step + self.step][9]  # 価格実績値の取得
        self.imbalance = self.input_data[self.day * self.total_step + self.step][10]  # インバランス料金の取得

        # 状態変数の初期化
        self.state = np.zeros(self.ACTION_NUM * 2 + 1)
        self.state[self.ACTION_NUM] = self.battery / self.battery_MAX

        return self.state

    def step(self, action):
        # アクションの取得
        self.step += 1

        # バッテリーの状態更新
        self.battery += (action * self.battery_MAX * self.PV_max) / self.total_step
        self.battery = np.clip(self.battery, 0, self.battery_MAX)  # バッテリー容量を制約範囲内に収める

        # データの更新
        self.PV = self.input_data[self.day * self.total_step + self.step][8]  # PV発電量の取得
        self.price_predict = self.predict_data[self.day * self.total_step + self.step][9]  # 価格予測値の取得
        self.price_real = self.input_data[self.day * self.total_step + self.step][9]  # 価格実績値の取得
        self.imbalance = self.input_data[self.day * self.total_step + self.step][10]  # インバランス料金の取得

        # 状態の更新
        self.state = np.roll(self.state, -1)
        self.state[-1] = action
        self.state[self.ACTION_NUM] = self.battery / self.battery_MAX

        # 報酬の計算
        self.reward = self.reward_function(action)

        # エピソード完了の判定
        if self.step >= self.total_step:
            self.done = True

        return self.state, self.reward, self.done, self.info

    def reward_function(self, action):
        reward = -self.price_predict * action  # 基本的な報酬（価格予測値×アクション）

        # インバランス料金の加算
        if action < 0:
            reward -= self.imbalance * abs(action)  # 買電時のみインバランス料金が発生

        return reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass

        