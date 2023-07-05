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

class ESS_Model(gym.Env):
    def __init__(self, mode, pdf_day, train_days, test_day, PV_parameter, action_space):
        #パラメータの定義
        self.episode = 0
        self.total_step = action_space # 1Dayの総コマ数
        self.gamma = ma.exp(-(1/action_space)) # 放電に対する割引率
        self.omega = ma.exp(1/action_space) # 充電に対する割引率
        self.battery_MAX = 4 # ４kWh
        self.MAX_reward = -10000
        self.Train_Days = train_days # 学習Day
        self.test_days = test_day - 1 # テストDay数
        self.mode = mode
        if mode == "train":
            self.last_day = self.Train_Days
        elif mode == "test":
            self.last_day = self.test_days
        self.all_rewards = []
        #データのロード
        print("-データロード-")
        #学習データ
        input_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
        #テストデータ(これが充放電計画策定したいもの)
        predict_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv")

        #学習データの日数+1日分データが必要
        #空データドッキング
        data = [[0] * 20] * 48
        columns = ["year","month","day","hour","temperature","total precipitation","u-component of wind","v-component of wind","radiation flux","pressure","relative humidity","PVout","price","imbalance",
                   "yearSin","yearCos","monthSin","monthCos","hourSin","hourCos"]
        new_rows_df = pd.DataFrame(data, columns=columns)
        input_data = input_data.append(new_rows_df, ignore_index=True)

        #データの作成
        print("-データ作成-")
        if self.mode == "train":

            self.time_stamp = input_data["hour"]

            price_all = predict_data["price"]/2
            true_all_price = input_data["price"]/2
            imbalance_all = predict_data["imbalance"]/2
            true_imbalance_all = input_data["imbalance"]/2
            self.PV = PV_parameter #Upper, lower, PVoutの選択用
            PV_out_all = predict_data[self.PV]
            PV_true_all = input_data["PVout"]

            price_data = true_all_price
            price_true_data = true_all_price
            PV_out_data = PV_true_all
            PV_true_data = PV_true_all
            imbalance_all = true_imbalance_all
            true_imbalance_all = true_imbalance_all

        elif self.mode == "test":

            self.time_stamp = predict_data["hour"]

            price_all = predict_data["price"]/2
            true_all_price = predict_data["price"]/2
            imbalance_all = predict_data["imbalance"]/2
            true_imbalance_all = predict_data["imbalance"]/2
            #self.PV = PV_parameter #upper, lower, PVoutの選択用
            PV_out_all = predict_data["PVout"]
            PV_true_all = predict_data["PVout"]
        
            price_data = price_all
            price_true_data = price_all
            PV_out_data = PV_out_all
            PV_true_data = PV_out_all
            imbalance_all = imbalance_all
            true_imbalance_all = imbalance_all
           

        #pandas -> numpy変換,型変換
        print("-データ変換-")
        self.price_all = price_data.values
        self.price = self.price_all.reshape((len(self.price_all), 1)) 

        self.true_all_price = price_true_data.values
        self.true_price = self.true_all_price.reshape((len(self.true_all_price), 1)) 

        self.imbalance_all = imbalance_all.values
        self.imbalance = self.imbalance_all.reshape((len(self.imbalance_all), 1)) 

        self.true_imbalance_all = true_imbalance_all.values
        self.true_imbalance = self.true_imbalance_all.reshape((len(self.true_imbalance_all), 1))

        self.PV_out_all = PV_out_data.values
        self.PV_out = self.PV_out_all.reshape((len(self.PV_out_all), 1))
        

        self.PV_true_all = PV_true_data.values
        self.PV_true = self.PV_true_all.reshape((len(self.PV_true_all), 1)) 
        

        #アクション
        self.ACTION_NUM = action_space #アクションの数(現状は48の約数のみ)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape = (self.ACTION_NUM,))
        #状態の上限と下限の設定
        low_box = np.zeros(self.ACTION_NUM*2+1) # 入力データの下限値×入力データの数
        high_box = np.ones(self.ACTION_NUM*2+1) # 入力データの上限値×入力データの数
        LOW = np.array(low_box)
        HIGH = np.array(high_box)
        self.observation_space  = gym.spaces.Box(low=LOW, high=HIGH)
        # 初期データの設定
        self.reset()

    def step(self, action): # rewardの決定
        done = False # True:終了　False:学習継続
        reward = 0
        all_action = action
        #action > 0 →放電  action < 0 →充電
        for self.time_stamp in range(0, self.ACTION_NUM):

            action = float(all_action[self.time_stamp])
            ACTION = action*1.5
            ACTION = round(ACTION, 1)
            time = self.time
            count = self.count
            soc = (self.battery / self.battery_MAX) # %
            soc_true = (self.battery_true / self.battery_MAX)
            battery_true = self.battery_true

            self.all_soc.append(soc*100)
            self.all_battery.append(self.battery)
            self.all_soc_true.append(soc_true*100)
            self.all_price.append(self.price_time)
            self.all_price_true.append(self.true_price_time)
            self.all_time.append(time/2)
            self.all_count.append(count/2)
            self.all_action.append(ACTION)
            self.all_PV_out_time.append(self.PV_out_time[0])
            self.all_PV_true_time.append(self.PV_true_time[0])
            self.all_imbalance.append(self.imbalance)
            self.all_imbalance_true.append(self.true_imbalance)

            if self.PV_out_time < 0:
                self.PV_out_time = [0]
                    
            if self.PV_out_time < -ACTION and action < 0:
                action_real = -self.PV_out_time
            elif action > 0 and 0 < self.battery < ACTION:
                action_real = self.battery
            elif self.battery == self.battery_MAX and action < 0:
                action_real = 0
            elif action > 0 and self.battery == 0:
                action_real = 0
            else:
                action_real = ACTION
            self.all_action_fil.append(action_real)

            if self.PV_true_time < -ACTION and action < 0:
                action_true = -self.PV_true_time[0]
            elif action > 0 and 0 < battery_true < ACTION:
                action_true = battery_true
            elif battery_true == self.battery_MAX and action < 0:
                action_true = 0
            elif action > 0 and battery_true == 0:
                action_true = 0
            else:
                action_true = ACTION
            self.all_action_true.append(action_true)

            pred_battery = self.battery
            pred_battery_true = battery_true
            next_battery = self.battery - action_real/2
            next_battery = next_battery
            battery_true = battery_true - action_true/2
            battery_true = battery_true
            if battery_true < 0:
                battery_true = 0
            elif battery_true > self.battery_MAX:
                battery_true = np.array([self.battery_MAX])
                battery_true = battery_true[0]
            if next_battery > self.battery_MAX:
                next_battery = self.battery_MAX
            elif next_battery < 0:
                next_battery = 0
            if action_real < 0:
                self.PV_out_time = self.PV_out_time - (self.battery - pred_battery) # 充電に使った分を引く
            if action_true < 0:
                self.PV_true_time = self.PV_true_time - (battery_true - pred_battery_true) # 充電に使った分を引く

            n_battery = self.battery - ACTION/2 # 評価用の充電残量

            # ---Reward設定---------------
            reward += self.reward_set(ACTION ,n_battery)
            # --------------------------
                            
            self.time += 1
            time = self.time
            self.count += 1
            self.battery = next_battery
            self.battery_true = battery_true
            soc = (self.battery / self.battery_MAX) # %
            soc_true = (self.battery_true / self.battery_MAX) # %

            if self.time == 48:
                self.days += 1
                self.time = 0

            self.sell_PVout.append(self.PV_out_time[0])
            self.sell_PVtrue.append(self.PV_true_time[0])
            # 入力データ(学習時：実測　テスト時：予測)
            self.data_set()
        
        self.rewards.append(reward)
        if time == 48 and self.days == self.last_day and self.mode == "train": #学習の経過表示、リセット
            if self.episode == 0:
                self.MAX_reward = np.sum(self.rewards)
            self.episode += 1

            print("episode:"+str(self.episode) + "/"+str(episode))

            self.all_rewards.append(np.sum(self.rewards))

            if np.sum(self.rewards) >= self.MAX_reward:
                self.MAX_reward = np.sum(self.rewards) # rewardの最高値
                self.evalution("Battery-Control-By-Reinforcement-Learning/" + "result-" + self.mode + ".pdf")
                self.model.save("ESS_model")
                self.end_count = 0
            elif np.sum(self.rewards) < self.MAX_reward:
                self.end_count += 1

            if self.end_count >= 20000:
                if self.episode == 100000 or self.episode > 20000:
                    self.evalution("Battery-Control-By-Reinforcement-Learning/" + "result-" + self.mode + "-end.pdf")
                    self.model.save("ESS_model_end")
                    #done = True # 学習終了
                    self.end_count = 0

        if time == 48 and self.days == self.last_day and self.mode == "test":
            self.evalution("Battery-Control-By-Reinforcement-Learning/" + "result-" + self.mode + ".pdf")

        if time == 48 and self.days == self.last_day:
            state = self.reset()
        else:
            state = [soc]
            state.extend(self.input_PV_data)
            state.extend(self.input_price_data)

        return state, reward, done, {}
    
    def reset(self): # 状態を初期化
        self.time = 0
        self.count = 0
        self.battery = 0
        self.battery_true = 0 
        self.days = 1
        self.rewards = []
        self.all_PV_out_time = []
        self.all_PV_true_time = []
        self.all_soc = []
        self.all_soc_true = []
        self.all_soc_real = []
        self.all_battery = []
        self.all_price = []
        self.all_price_true = []
        self.all_time = []
        self.all_count = []
        self.all_action = []
        self.all_action_fil = []
        self.all_action_true = []
        self.all_imbalance = []
        self.all_imbalance_true = []
        self.sell_PVout = []
        self.sell_PVtrue = []

        self.data_set()
        state = [self.battery/4]
        state.extend(self.input_PV_data)
        state.extend(self.input_price_data)

        return state

    # --------使わないけど必要---------------
    def render(self, mode='human', close=False):
        pass

    def close(self): 
        pass

    def seed(self): 
        pass
    # ---------------------------------------

    # 入力データの設定
    def data_set(self):
        
        self.PV_out_time = self.PV_out[self.time]
        self.PV_true_time = self.PV_true[self.time]
        self.price_time = self.price[self.time]
        self.true_price_time = self.true_price[self.time]
        self.imbalance_time = self.imbalance[self.time]
        self.true_imbalance_time = self.true_imbalance[self.time]
        
        if self.mode == "train":
            if self.days != self.last_day:
                self.MAX_price = max(self.true_price[48*(self.days - 1):48*self.days])
            self.input_price = self.true_price[48*(self.days - 1) + self.time]
            self.input_PV = self.PV_true[48*(self.days - 1) + self.time]
            self.input_PV_data = (self.PV_true[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/2).T[0]
            self.input_price_data = (self.true_price[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/self.MAX_price).T[0]
            self.input_imbalance_data = (self.true_imbalance[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/self.MAX_price).T[0]
        elif self.mode == "test":
            if self.days != self.last_day:
                self.MAX_price = max(self.price)
            self.input_PV = self.PV_out[self.time]
            self.input_price = self.price[self.time]
            self.input_PV_data = (self.PV_out[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/2).T[0]
            self.input_price_data = (self.price[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/self.MAX_price).T[0]
            self.input_imbalance_data = (self.imbalance[48*(self.days - 1) + self.time:48*(self.days - 1) + self.time + self.ACTION_NUM]/self.MAX_price).T[0]

    #Reward設定
    def reward_set(self, ACTION, n_battery):
        reward = 0
        # 現在の状態と行動に対するreward
        if ACTION <= 0:
            if -ACTION > self.input_PV: # 充電する量がPV出力より高いならペナルティ(今の状態×行動)
                reward += ((self.omega)**(self.time_stamp))*self.input_price*ACTION
        
        if ACTION > 0:
            if ACTION > self.battery: # 放電量がSoCより大きいならペナルティ(今の状態×行動)
                reward += ((self.omega)**(self.time_stamp))*self.input_price*(self.battery - ACTION)
            if ACTION <= self.battery:# 放電のときのreward(今の状態×行動)
                reward += ((self.gamma)**(self.time_stamp))*self.input_price*ACTION

        # 次の状態と行動に対するreward
        if n_battery > self.battery_MAX: # SoCが100％以上でペナルティ
            reward += ((self.omega)**(self.time_stamp))*self.input_price*(-n_battery)

        return reward

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
episode = 100 # 10000000  

print("--Trainモード開始--")

# test 1Day　Reward最大
pdf_day = 0 #確率密度関数作成用のDay数 75 80
train_days = 366 # 学習Day数 70 ~ 73
test_day = 3 # テストDay数 + 2 (最大89)
PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類)　#今後はUpper, lower, PVout
mode = "train" # train or test
model_name = "ESS_model" # ESS_model ESS_model_end

# Training環境設定と実行
env = ESS_Model(mode, pdf_day, train_days, test_day, PV_parameter, action_space)
env.main_root(mode, num_episodes, train_days, episode, model_name)# Trainingを実行

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
