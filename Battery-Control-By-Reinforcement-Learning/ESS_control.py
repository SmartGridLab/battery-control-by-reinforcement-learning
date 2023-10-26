import gym
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math as ma
import tkinter as tk
#import tensorflow as tf

from matplotlib.backends.backend_pdf import PdfPages
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter # tensorBoardを起動して、学習状況を確認する

print("\n---充放電計画策定プログラム開始---\n")

warnings.simplefilter('ignore')

#グラフ作成 評価値算出
def evalution(self, pdf_name):
    pp = PdfPages(pdf_name) # PDFの作成
    if self.mode == "train":
        graph_1 = self.graph(self.all_rewards)
        pp.savefig(graph_1)
    graph_2 = self.schedule(self.all_action,self.all_PV_true_time,self.all_soc, mode = 0)
    graph_3 = self.schedule(self.all_action,self.all_PV_true_time,self.all_soc, mode = 1)
    graph_4 = self.schedule(self.all_action_true,self.all_PV_true_time,self.all_soc_true, mode = 0)
    graph_5 = self.schedule(self.all_action_true,self.all_PV_true_time,self.all_soc_true, mode = 1)
    pp.savefig(graph_2)
    pp.savefig(graph_3)
    pp.savefig(graph_4)
    pp.savefig(graph_5)
    if self.mode == "test":
        graph_6 = self.schedule_PV(self.all_PV_true_time,self.all_PV_out_time)
        pp.savefig(graph_6)
        self.imb_eva() # インバランス料金の算出(評価値)
        pp.savefig(self.imb_Graph)
        pp.savefig(self.imb_Graph_PV)
    pp.close()

def schedule(self, action, PV, soc, mode):
    fig = plt.figure(figsize=(22, 12), dpi=80)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax2.set_ylim([-1,101])
    ax1.tick_params(axis='x', labelsize=35)
    ax1.tick_params(axis='y', labelsize=35)
    ax2.tick_params(axis='x', labelsize=35)
    ax2.tick_params(axis='y', labelsize=35)
    if self.mode == "train":
        ax1.plot(self.all_time, action, "blue", drawstyle="steps-post",label="Charge and discharge")
        ax1.plot(self.all_time, PV, "Magenta",label="PV generation")
        ax2.plot(self.all_time, soc, "red",label="SoC")
    elif self.mode == "test":
        ax1.plot(self.all_count, action, "blue", drawstyle="steps-post",label="Charge and discharge")
        ax1.plot(self.all_count, PV, "Magenta",label="PV generation")
        ax2.plot(self.all_count, soc, "red",label="SoC")
    if mode == 0: # 電力価格ありのグラフ
        if self.mode == "train":
            ax1.plot(self.all_time, self.all_price_true, "green",drawstyle="steps-post",label="Power rates")
        elif self.mode == "test":
            ax1.plot(self.all_count, self.all_price_true, "green",drawstyle="steps-post",label="Power rates")
        ax1.set_ylabel("Power [kW] / Power rates [Yen]", fontsize = 35)
    elif mode == 1:
        ax1.set_ylim([-2,2])
        ax1.set_ylabel("Power [kW]", fontsize = 35)    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', prop={"size": 35}).get_frame().set_alpha(0.0)
    if self.mode == "train":
        ax1.set_xlim([0,23.5])
    elif self.mode == "test":
        ax1.set_xlim([0,23.5*(self.test_days - 1)])
    ax1.set_xlabel('Time [hour]', fontsize = 35)
    ax1.grid(True)
    ax2.set_ylabel("SoC[%]", fontsize = 35)
    plt.tick_params(labelsize=35)
    plt.close()

    if self.mode == "test":
        self.all_count = pd.DataFrame(self.all_count)
        action = pd.DataFrame(action)
        soc = pd.DataFrame(soc)
        PV = pd.DataFrame(PV)
        price = pd.DataFrame(self.all_price_true)
        result_data = pd.concat([self.all_count,action],axis=1)
        result_data = pd.concat([result_data,PV],axis=1)
        result_data = pd.concat([result_data,soc],axis=1)
        result_data = pd.concat([result_data,price],axis=1)
        label_name = ["hour","charge/discharge","PVout","soc","price"] # 列名
        result_data.columns = label_name # 列名付与
        result_data.to_csv("Battery-Control-By-Reinforcement-Learning/result_data.csv")

    return fig

def schedule_PV(self, PV_true, PV_pred):
    fig = plt.figure(figsize=(22, 12), dpi=80)
    ax1 = fig.add_subplot(111)
    #ax2 = ax1.twinx()
    #ax2.set_ylim([-1,101])
    ax1.tick_params(axis='x', labelsize=35)
    ax1.tick_params(axis='y', labelsize=35)
    #ax2.tick_params(axis='x', labelsize=35)
    #ax2.tick_params(axis='y', labelsize=35)
    #ax1.plot(self.all_count, action, "blue", drawstyle="steps-post",label="充放電")
    ax1.plot(self.all_count, PV_true, "red",label="PV generation: Actual")
    ax1.plot(self.all_count, PV_pred, "blue",label="PV generation: Forecast")
    #ax2.plot(self.all_count, soc, "red",label="SoC")
    ax1.plot(self.all_count, self.all_price_true, "green",drawstyle="steps-post",label="Electricity Price: Actual")
    ax1.plot(self.all_count, self.all_price, "Magenta",drawstyle="steps-post",label="Electricity Price: Forecast")
    ax1.set_ylabel("Power [kW] Price [Yen]", fontsize = 35) 
    h1, l1 = ax1.get_legend_handles_labels()
    #h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='upper left', prop={"size": 35}).get_frame().set_alpha(0.0)
    ax1.set_xlim([0,23.5*(self.test_days - 1)])
    ax1.set_xlabel('Time [hour]', fontsize = 35)
    ax1.grid(True)
    #ax2.set_ylabel("SoC[%]", fontsize = 35)
    plt.tick_params(labelsize=35)
    plt.close()

    return fig

def graph(self, y):
    fig = plt.figure(figsize=(24, 14), dpi=80)
    plt.plot(np.arange(self.episode), y, label = "Reward")
    plt.legend(prop={"size": 35})
    plt.xlabel("Episode", fontsize = 35)
    plt.ylabel("Reward", fontsize = 35)
    plt.tick_params(labelsize=35)
    plt.close()
    
    return fig

def imb_eva(self):
    self.all_action_true = pd.DataFrame(np.ravel(self.all_action_true))
    self.all_action_fil = pd.DataFrame(np.ravel(self.all_action_fil))
    self.sell_PVout = pd.DataFrame(np.ravel(self.sell_PVout))
    self.sell_PVtrue = pd.DataFrame(np.ravel(self.sell_PVtrue))
    self.imbalance = pd.DataFrame(np.ravel(self.all_imbalance))
    #self.imbalance_true = pd.DataFrame(np.ravel(self.all_imbalance_true))

    # インバランス料金、利益等の算出(評価値の算出)
    print("-評価値算出-")
    imbalance = 0
    total_profit = 0
    profit = 0
    imbalance_PV = 0
    PV_profit_true = 0
    imb_all = []
    sell_all = []
    imb_PV = []
    sell_PV = []
    x = 0
    for i in range(0, len(self.all_action_true)):
        x += 1
        true_PV_forecast_time = self.sell_PVtrue[0][i]
        true_ESS_forecast_time = self.all_action_true[0][i]
        Forecast_ESS_time = self.all_action_fil[0][i]
        Forecast_PV_time = self.sell_PVout[0][i]
        imbalance_price = self.all_imbalance[0][i]
        #beta = self.all_beta[0][i]
        price = self.true_price[i]
        PVtrue = self.all_PV_true_time[i]
        PVout = self.all_PV_out_time[i]

        if true_ESS_forecast_time > 0:
            true_total_forecast_time = true_PV_forecast_time + true_ESS_forecast_time
        elif true_ESS_forecast_time <= 0:
            true_total_forecast_time = true_PV_forecast_time
        if Forecast_ESS_time > 0:
            total_forecast_time_real = Forecast_PV_time + Forecast_ESS_time
        elif Forecast_ESS_time <= 0:
            total_forecast_time_real = Forecast_PV_time
        total_profit += true_total_forecast_time*price # PV＋ESSによる売上

        PV_profit_true += PVtrue*price # PV実測のみによる売上

        # 不足
        if true_total_forecast_time < total_forecast_time_real:     
            ###############インバランス料金変更#################       
            imbalance -= imbalance_price*(abs(true_total_forecast_time - total_forecast_time_real))
        # 余剰
        elif true_total_forecast_time > total_forecast_time_real:
            ###############インバランス料金変更#################
            imbalance -= imbalance_price*(abs(true_total_forecast_time - total_forecast_time_real))
        elif true_total_forecast_time == total_forecast_time_real:
            imbalance -= 0
            
        # 不足
        if PVout < PVtrue:
            ###############インバランス料金変更#################                
            imbalance_PV -= imbalance_price*(abs(PVtrue - PVout))
        # 余剰
        elif PVout > PVtrue:
            ###############インバランス料金変更#################
            imbalance_PV -= imbalance_price*(abs(PVtrue - PVout))
        elif PVout == PVtrue:
            imbalance_PV -= 0

        if x == 48:#24:00の処理
            imb_all.append(imbalance)
            sell_all.append(total_profit[0])
            sell_PV.append(PV_profit_true[0])
            imb_PV.append(imbalance_PV)
            x = 0

    sell_all = np.array(sell_all)
    imb_all = np.array(imb_all)
    sell_PV = np.array(sell_PV)
    imb_PV = np.array(imb_PV)
    plus = sell_all+imb_all
    plus_PV = sell_PV+imb_PV
    self.imb_Graph = self.graph_imb(sell_all,imb_all,plus,"Day", "Yen", "Sales","Imbalance Penalty","Profit")
    self.imb_Graph_PV = self.graph_imb(sell_PV,imb_PV,plus_PV,"Day", "Yen", "Sales","Imbalance Penalty","Profit")
    print("PV+ESS")
    print(total_profit)
    print((-1)*imbalance)
    print(total_profit + imbalance)
    print("PV")
    print(PV_profit_true)
    print((-1)*imbalance_PV)
    print(PV_profit_true + imbalance_PV)

def graph_imb(self,y1, y2, y3, x_label, y_label, label_name_1, label_name_2, label_name_3):
    episode = len(self.all_action_true)/48 # テストDay数
    fig = plt.figure(figsize=(24, 10), dpi=80)
    ax = fig.add_subplot(111)
    ax.set_ylim(-2000, 4000)
    plt.rcParams["font.size"] = 35
    plt.plot(np.arange(episode), y1, label = label_name_1)
    plt.plot(np.arange(episode), y2, label = label_name_2)
    plt.plot(np.arange(episode), y3, label = label_name_3)
    plt.legend(fontsize = 35)
    plt.xlabel(x_label, fontsize = 35)
    plt.ylabel(y_label, fontsize = 35)
    plt.close()
        
    return fig

#メインルーチン   
#root = tk.Tk()
#root.mainloop()
def main_root(self, mode, num_episodes, train_days, episode, model_name):
    
    # #Tkinter処理 epsode途中に終了を防ぐ
    # root = tk.Tk()
    # root.withdraw()
    
    if mode == "train":
        print("-モデル学習開始-")
        #self.model = PPO("MlpPolicy", env, gamma = 0.9, verbose=0, ent_coef = 0.01, learning_rate = 0.0001, n_steps = 48, tensorboard_log="./PPO_tensorboard/") 
        self.model = PPO("MlpPolicy", env, gamma = 0.8, gae_lambda = 1, clip_range = 0.2, 
                        ent_coef = 0.005, vf_coef =0.5, learning_rate = 0.0001, n_steps = 48, 
                        verbose=0, tensorboard_log="./PPO_tensorboard/") 
        #モデルの学習
        self.model.learn(total_timesteps=num_episodes*train_days*episode)
        print("-モデル学習終了-")

    
    if mode == "test":
        #モデルのロード
        print("-モデルロード-")
        self.model = PPO.load(model_name)
        #モデルのテスト
        obs = env.reset() # 最初のstate
        obs = pd.Series(obs)
        obs = torch.tensor(obs.values.astype(np.float64))
        for i in range(0, num_episodes*(self.test_days - 1)):
            action, _ = self.model.predict(obs)
            obs, reward, done, _ = self.step(action)
            obs = pd.Series(obs)
            obs = torch.tensor(obs.values.astype(np.float64))

#30分1コマで、何時間先まで考慮するか
action_space = 12 #アクションの数(現状は48の約数のみ)
num_episodes = int(48/action_space) # 1Dayのコマ数(固定)

# 学習回数
episode = 1000 # 10000000  

print("--Trainモード開始--")

# test 1Day　Reward最大
pdf_day = 0 #確率密度関数作成用のDay数 75 80
train_days = 366 # 学習Day数 70 ~ 73
test_day = 33 # テストDay数 + 2 (最大89)
PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類)　#今後はUpper, lower, PVout
mode = "train" # train or test
model_name = "ESS_model" # ESS_model ESS_model_end

# Training環境設定と実行
#env = ESS_Model(mode, pdf_day, train_days, test_day, PV_parameter, action_space)
#env.main_root(mode, num_episodes, train_days, episode, model_name)# Trainingを実行

print("--Trainモード終了--")



print("--充放電計画策定開始--")

# test 1Day　Reward最大
pdf_day = 0 #確率密度関数作成用のDay数 75 80
train_days = 366 # 学習Day数 70 ~ 73
test_day = 33 # テストDay数 + 2 (最大89)
PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類) #今後はUpper, lower, PVout
mode = "test" # train or test
model_name = "ESS_model" # ESS_model ESS_model_end

# Test環境設定と実行 学習
env = ESS_Model(mode, pdf_day, train_days, test_day, PV_parameter, action_space)
env.main_root(mode, num_episodes, train_days, episode, model_name)

print("--充放電計画策定終了--")


print("\n---充放電計画策定プログラム終了---\n")

"""
# Test環境のパラメータ設定の種類:

# test １Day　Reward最大
pdf_day = 76 #確率密度関数作成用のDay数 75 80
train_days = 30 # 学習Day数 70 ~ 73
test_day = 3 # テストDay数 + 2 (最大89)
PV_parameter = "Forecast" # Forecast or PVout_true (学習に使用するPV出力値の種類)
mode = "test" # train or test
model_name = "ESS_model" # ESS_model ESS_model_end

# test 2カ月 学習終了
pdf_day = 59 #確率密度関数作成用のDay数
train_days = 30 # 学習Day数
test_day = 30 # テストDay数 + 2 (最大89)
PV_parameter = "Forecast" # Forecast or PVout_true (学習に使用するPV出力値の種類)
mode = "test" # train or test
model_name = "ESS_model_end" # ESS_model ESS_model_end

# test 1Day 学習終了
pdf_day = 75 #確率密度関数作成用のDay数
train_days = 30 # 学習Day数 70 ~ 73
test_day = 4 # テストDay数 + 2 (最大89)
PV_parameter = "Forecast" # Forecast or PVout_true (学習に使用するPV出力値の種類)
mode = "test" # train or test
model_name = "ESS_model_end" # ESS_model ESS_model_end
"""
