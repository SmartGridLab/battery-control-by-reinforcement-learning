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
from RL_env import ESS_ModelEnv
from RL_train import TrainModel
from RL_test import TestModel

warnings.simplefilter('ignore')
print("\n---充放電計画策定プログラム開始---\n")

# 実行部分
if __name__ == "__main__" :
    # 学習回数
    episode = 6 # 10000000

    print("--Training開始--")

    # test 1Day　Reward最大
    pdf_day = 0 #確率密度関数作成用のDay数 75 80
    train_days = 366 # 学習Day数 70 ~ 73
    test_day = 3 # テストDay数 + 2 (最大89)
    PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類)　#今後はUpper, lower, PVout
    model_name = "ESS_model" # ESS_model ESS_model_end

    # Training環境設定と実行
    env = ESS_ModelEnv(pdf_day, train_days, test_day, PV_parameter)
    env.train_PPOModel(train_days, episode, model_name) # Trainingを実行

    print("--Trainモード終了--")

    print("--充放電計画策定開始--")

    # test 1Day　Reward最大
    pdf_day = 0 #確率密度関数作成用のDay数 75 80
    train_days = 366 # 学習Day数 70 ~ 73
    test_day = 3 # テストDay数 + 2 (最大89)
    PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類) #今後はUpper, lower, PVout
    model_name = "ESS_model" # ESS_model ESS_model_end

    # Test環境設定と実行 学習
    env = ESS_Model(pdf_day, train_days, test_day, PV_parameter)
    env.main_root(mode, train_days, episode, model_name)

    print("--充放電計画策定終了--")
    print("\n---充放電計画策定プログラム終了---\n")


    
