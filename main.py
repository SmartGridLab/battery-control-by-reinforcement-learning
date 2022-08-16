#Import 
import gym
import warnings
import numpy as np
import pandas as pd
import math as ma
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# Import thirdparty modules
from matplotlib.backends.backend_pdf import PdfPages
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import A2C, PPO2
from gym import spaces
from scipy.stats import norm
# Import locally developed modules
import ESSmodel
import getGraph

warnings.simplefilter('ignore')

#メインルーチン    
def main_root(self, mode, num_episodes, train_days, episode, model_name):
    if mode == "learn":
        self.model = PPO2("MlpPolicy", env, gamma = 0.9, verbose=0, learning_rate = 0.0001, n_steps = 48) # モデルの定義(A2C) 

        #モデルの学習
        # model.learnを呼び出すときに、自動的に最初にmodelのdef reset(self):が呼び出される
        # 明示的にresetが呼び出されることはClass ESSmodelには書かれていない
        self.model.learn(total_timesteps=num_episodes*train_days*episode)
    
    if mode == "test":
        #モデルのロード
        self.model_name = model_name
        self.model = PPO2.load(model_name)
        #モデルのテスト
        obs = env.reset() # 最初のstate
        for i in range(0, num_episodes*(self.test_days - 1)):
            action, _ = self.model.predict(obs)
            obs, reward, done, _ = self.step(action)
print("end")

#パラメータ(学習条件などは以下のパラメータを変更するだけで良い)
num_episodes = 48 # 1日のコマ数(固定)
pdf_day = 0 #確率密度関数作成用の日数を指定する
train_days = 10 # 学習日数
test_day = 6 # テスト日数＋１
episode = 100  # 学習回数
PV_parameter = "PVout_true" # Forecast or PVout_true (学習に使用するPV出力値の種類)
mode = "learn" # learn or test
model_name = "ESS_learn_1000" # ESS_learn ESS_learn_1000

env = ESSmodel(mode, pdf_day, train_days, test_day, PV_parameter) # 環境設定
env.main_root(mode, num_episodes, train_days, episode, model_name) # Training or Testsを実行