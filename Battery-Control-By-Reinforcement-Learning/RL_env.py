# インポート：外部モジュール
import gym
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import math
import tkinter as tk
from matplotlib.backends.backend_pdf import PdfPages
from stable_baselines3 import PPO
#from torch.utils.tensorboard import SummaryWriter # tensorBoardを起動して、学習状況を確認する

# internal modules
import RL_visualize as visual
import RL_dataframe_manager as df_manage

warnings.simplefilter('ignore')

class ESS_ModelEnv(gym.Env):
    def __init__(self, mode, train_days, test_days):      
        # PPOで使うパラメーターの設定   
        self.gamma = math.exp(-(1/action_space)) # 放電に対する割引率
        self.omega = math.exp(1/action_space) # 充電に対する割引率
        self.reward_range = (-10000, math.inf)  # Reardの範囲

        # Batteryのパラメーター
        self.battery_max_cap = 4 # 蓄電池の定格容量4kWh
        self.battery_current_energy = 0 # SoC[kW]初期値

        # 学習とテストに使うデータの選択
        self.train_days = train_days # 学習する日数
        self.test_days = test_days # テストDay数

        # action spaceの定義(上下限値を設定。actionは連続値。)
        self.action_spcae = gym.spaces.Box(low=-1.0, high=1.0) 
        # 状態の上限と下限の設定
        self.observation_space  = gym.spaces.Box(low=0, high=1)

        # step関数内でつかうカウンター
        self.state_idx = 0 # time_steps in all episodes (time frames in train/test days: 48*n)
        self.total_reward = 0 # 全episodeでの合計のreward
        self.reward = [] # 各stepでのreward

        # Mode選択
        self.mode = mode    # train or test

    #### time_stepごとのactionの決定とrewardの計算を行う
    def step(self, action):
        # time_stepを一つ進める
        self.state_idx += 1
        # Socを計算
        soc = (self.battery_cunnret_energy / self.battery_max_cap) # SoC∈[0,1]へ変換

        #### rewardの計算
        _get_reward(df.input)
        # 評価用の充電残量
        n_battery = self.battery - action
        # これまでのrewardに時刻self.timeのrewardを加算
        reward += self.reward_set(action ,n_battery)
        # SoC算出
        self.battery = next_battery
        soc = (self.battery / self.battery_MAX) # %
        # timeが最終コマのとき
        if self.time == 48:
            self.days += 1
            self.time = 0
        # 売電量の更新
        energy_transfer = self.PV_out_time[0] * 0.5 #[kW]->[kWh]
        self.all_energy_transfer.append(energy_transfer)

        self.evalution("Battery-Control-By-Reinforcement-Learning/" + "result-" + self.mode + ".pdf")
        self.model.save("ESS_model")

        # checking whether our episode (day) ends
        if self.state_idx == 48
        done = False # True:終了　False:学習継続 -> Trueだと勝手にresetが走る


        # actionした結果をobservationにいれる
        observation = [soc]
        observation.extend(self.input_PV_data)
        observation.extend(self.input_price_data)
        
        # 付随情報をinfoに入れる
        info = {}
        
        ## time_stepを次へ進める
        if self.time == self.last_timestep:
            done = True
        else
            self.time_step += 1
        

        return observation, reward, done, info
    
    # 状態の初期化
    # - step関数において、done = Trueになると呼ばれる。任意のタイミングで呼ぶこともできる。
    # - 1episode(1日)終わるとresetが呼ばれるので、次の日のデータをstateへ入れる
    # - 現状ではQuantile 50%の予測を使っている -> 改良が必要。任意に選べる等
    # MPI: MPIの予測値
    # SSP: SSPの予測値
    # wind: MPIの予測値
    # PV: MPIの予測値
    def reset(self):
        state = [
            df_manage.df_input(MIP_q50[state_idx:state_idx + 48]),   # MIP
            df_manage.df_input(SSP_q50[state_idx:state_idx + 48]),   # SSP
            df_manage.df_input(wind_q50[state_idx:state_idx + 48]),  # wind
            df_manage.df_input(PV_q50[state_idx:state_idx + 48])   # PV
        ]
        return state

    # Envを描写する関数 -> 使っていない
    def render(self, mode='human', close=False):
        pass
    # Envを開放する関数 -> 使ってないが、使いたい（メモリを節約するため）
    def close(self): 
        pass
    # 乱数のシードを指定する関数 -> 使ってないが、使うべき？
    def seed(self): 
        pass

    # timeごとのrewardの計算 > rewardの設定内容
    # - rewardは1日(1 episode)ごとに合計される
    # - action > 0 →放電  action < 0 →充電
    def _get_reward(self, action, n_battery):
        # 現在の状態と行動に対するreward
        # rewardはすべて入出力[kW]*値段[JPY/30min]で計算(実際の報酬ではない)
        # 充電する場合
        if action <= 0:
            # 売電(PV出力-BT入力)に対するreward(今の状態×行動)
            if -action < self.input_PV:
                reward += ((self.omega)**(self.state_idx))*self.input_price*(self.PV_out_time + action)
            # BT入力がPV出力より高いならペナルティ(今の状態×行動)
            if -action > self.input_PV:
                reward += ((self.omega)**(self.state_idx))*self.input_price*action
        
        # 放電する場合
        if action > 0:
            # PV出力(売電)に対するreward
            reward += ((self.gamma)**(self.state_idx))*self.input_price*self.PV_out_time
            # BT出力がSoCより大きいならペナルティ(今の状態×行動)
            if action > self.battery: 
                reward += ((self.omega)**(self.state_idx))*self.input_price*(self.battery - action)
            # BT出力(売電)に対するreward(今の状態×行動)...PV出力に加えてBT出力が報酬として加算される
            if action <= self.battery:
                reward += ((self.gamma)**(self.state_idx))*self.input_price*action

        # 次の状態と行動に対するreward
        # SoCが100％以上でペナルティ
        if n_battery > self.battery_MAX: 
            reward += ((self.omega)**(self.state_idx))*self.input_price*(-n_battery)

        return reward

    def _get_possible_schedule(action):
        #### actionを適正化(充電をPVの出力があるときのみに変更)
        # PV発電量が0未満の場合、0に設定
        if self.PV_out_time < 0:
            self.PV_out_time = [0]
        # 充電時、PV発電量<充電量 の場合、充電量をPV出力値へ調整
        if self.PV_out_time < -action and action < 0:
            action_real = -self.PV_out_time
        # 放電時、放電量>蓄電池残量の場合、放電量を蓄電池残量へ調整
        elif action > 0 and 0 < self.battery < action:
            action_real = self.battery
        # 充電時、蓄電池残量が定格容量に達している場合、充電量を0へ調整
        elif self.battery == self.battery_MAX and action < 0:
            action_real = 0
        # 放電時、蓄電池残量が0の場合、放電量を0へ調整
        elif action > 0 and self.battery == 0:
            action_real = 0
        # 上記条件に当てはまらない場合、充放電量の調整は行わない
        else:
            action_real = action
        # 実際の充放電量をリストに追加
        self.all_action_real.append(action_real)

        #### 蓄電池残量の更新
        # 次のtimeにおける蓄電池残量を計算
        next_battery = self.battery - action_real*0.5 #action_real*0.5とすることで[kWh]へ変換

        ### 蓄電池残量の挙動チェック
        # 次のtimeにおける蓄電池残量が定格容量を超える場合、定格容量に制限
        if next_battery > self.battery_MAX:
            next_battery = self.battery_MAX
        # 次のtimeにおける蓄電池残量が0kwh未満の場合、0に制限
        elif next_battery < 0:
            next_battery = 0
        # 充電の場合、PV発電量から充電量を差し引く
        if action_real < 0:
            self.PV_out_time = self.PV_out_time + action_real
