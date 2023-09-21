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

# モデルのトレーニング
def train(self, mode, num_act_window, train_days, episode, model_name):
    # [2023.09.07 小平の理解は以下の通り]
    # action_space: 何コマ先までの充放電を考慮するか
    # steps: 最適運用スケジュールを行う期間に対応（48だと48コマ分の充放電スケジュールを作成する）
    # episode: steps分のコマ数を１episodeとして、何回試すか（ランダム要素を評価するため）
    # num_act_window: action_space(12コマ)num_act_window(4batch)行って、1日48コマに相当する（１日分で1episode)
    print("-モデル学習開始-")
    self.model = PPO("MlpPolicy", env, gamma = 0.8, gae_lambda = 1, clip_range = 0.2, 
                    ent_coef = 0.005, vf_coef =0.5, learning_rate = 0.0001, n_steps = 48, 
                    verbose=0, tensorboard_log="./PPO_tensorboard/") 
    self.model.learn(total_timesteps=num_act_window*train_days*episode)
    print("-モデル学習終了-")

# モデルのテスト
def test(self, mode, num_act_window, train_days, episode, model_name):
    # 学習したモデルをロード
    self.model = PPO.load(model_name)
    # 学習したモデルをテスト
    obs = env.reset() # stateをリセットする
    obs = pd.Series(obs)
    obs = torch.tensor(obs.values.astype(np.float64))
    for i in range(0, num_act_window*(self.test_days - 1)):
        action, _ = self.model.predict(obs)
        obs, reward, done, _ = self.step(action)
        obs = pd.Series(obs)
        obs = torch.tensor(obs.values.astype(np.float64))

# 入力データの設定
def data_set(self):
    self.PV_out_time = self.PVout[self.time]
    self.price_time = self.price[self.time]
    self.imbalance_time = self.imbalance[self.time]
    
    if self.mode == "train":
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

    elif self.mode == "test":
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


# 実行部分
if __name__ == "__main__" :
    # 30分1コマで、何時間先まで考慮するか
    # action_spaceで指定されたコマ数の充放電を定め、結果のrewardの合計を求め、それを最大化するように学習する
    action_space = 12 #アクションの数(現状は48の約数のみ)
    num_act_window = int(48/action_space) # 1Dayのコマ数(48)/actiuon_space(12コマ)でaction_windowの数が求まる

    # 学習回数
    episode = 6 # 10000000

    print("--Training開始--")

    # test 1Day　Reward最大
    pdf_day = 0 #確率密度関数作成用のDay数 75 80
    train_days = 366 # 学習Day数 70 ~ 73
    test_day = 3 # テストDay数 + 2 (最大89)
    PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類)　#今後はUpper, lower, PVout
    mode = "train" # train or test
    model_name = "ESS_model" # ESS_model ESS_model_end

    # Training環境設定と実行
    env = ESS_Model(mode, pdf_day, train_days, test_day, PV_parameter, action_space)
    env.main_root(mode, num_act_window, train_days, episode, model_name) # Trainingを実行

    print("--Trainモード終了--")

    print("--充放電計画策定開始--")

    # test 1Day　Reward最大
    pdf_day = 0 #確率密度関数作成用のDay数 75 80
    train_days = 366 # 学習Day数 70 ~ 73
    test_day = 3 # テストDay数 + 2 (最大89)
    PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類) #今後はUpper, lower, PVout
    mode = "test" # train or test
    model_name = "ESS_model" # ESS_model ESS_model_end

    # Test環境設定と実行 学習
    env = ESS_Model(mode, pdf_day, train_days, test_day, PV_parameter, action_space)
    env.main_root(mode, num_act_window, train_days, episode, model_name)

    print("--充放電計画策定終了--")
    print("\n---充放電計画策定プログラム終了---\n")
