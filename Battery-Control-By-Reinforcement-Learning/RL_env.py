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
    def __init__(self, train_days, test_days):
        ## データ読み込み
        # 学習データの読み込み（予測日の前日までを学習データとして利用する）
        # train_days: 学習日数
        # target_day: 予測日
        self.df_input = df_manage.get_input_df(target_day, train_days)
        # テストデータの読み込み
        self.df_predict = df_manage.get_preidct_df(target_day, test_days)
        # 結果を格納するテーブルの読み込み
        self.df_result = df_manage.get_result_df(test_days)

        # Batteryのパラメーター
        self.battery_max_cap = 4 # 蓄電池の最大容量 ex.4kWh
        self.inverter_max_cap = 4 # インバーターの定格容量 ex.4kW
        self.obs_list = [0.5] # SoC[0,1]の初期値 ex.0.5 (50%)

        ## PPOで使うパラメーターの設定
        # action spaceの定義(上下限値を設定。actionは連続値。)
        # - 1time_step(ex.30 min)での充放電量(規格値[0,1])の上下限値を設定
        # - 本当はinverter_max_capとbattery_max_capを使って、上下限値を設定したい
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0)
        # 状態(observation=SoC)の上限と下限の設定
        self.observation_space  = gym.spaces.Box(low=0, high=1)

        # # Rewardの範囲(いらないかも)
        # self.reward_range = (-10000, math.inf) 

        # step関数内でつかうカウンター
        self.state_idx = 0 # time_steps in all episodes (all episodes is a sum of time frames in train/test days 48*days)
        self.reward_total = 0 # 全episodeでの合計のreward
        self.reward_list = [] # 各stepでのreward

        # # Mode選択
        # self.mode = train    # train or test

    #### time_stepごとのactionの決定とrewardの計算を行う
    def step(self, action):
        # time_stepを一つ進める
        self.state_idx += 1

        ## rewardの計算
        # - 各stepでのrewardをリストに追加
        # - actionは規格値[0,1]なので、battery_max_capをかけて、実際の充放電量[MhW or kWh]に変換する
        # - actionと最新のSoCの値を渡す
        # -----------------------------------------------------------------------------------------------------------------
        # Rewardは、時系列的に後ろの方になるほど係数で小さくする必要がある。1 episode内で後ろのsteoのrewardを小さくする実装を考える
        # _get_rewardからの戻りrewardにgammaとstate_idxをかければ良さそう。あとで　実装する。
        # ------------------------------------------------------------------------------------------------------------------
        reward = self._get_reward(action*self.inverter_max_cap, self.obs_list[-1]*self.battery_max_cap)  
        self.reward_list.append(reward)
        # 全episodeでのrewardを計算
        self.reward_total += self.reward_list[-1]
        # SoC[0,1]をactionの分だけ更新する
        # - obs_listの最後の要素(前timestepのSoC)にactionを足す
        obs = self.obs_list[-1] + action
        # 各timestepでのobsをリストに追加
        self.obs_list.append(obs)
        # モデルを保存する
        self.model.save("ESS_model")

        # checking whether our episode (day) ends
        # - 1日(1 episode)が終わったら、done = Trueにする
        # state_idxは48コマ(1日)で割った余りが0になると、1日終了とする
        if self.state_idx % 48 == 0:
            done = True # Trueだと勝手にresetが走る
        else:
            done = False         
        
        # 付随情報をinfoに入れる
        info = {}
        
        return obs, reward, done, info
    
    ## 状態の初期化
    # - step関数において、done = Trueになると呼ばれる。任意のタイミングで呼ぶこともできる。
    # - 1episode(1日)終わるとresetが呼ばれるので、次の日のデータをstateへ入れる
    # - 現状ではQuantile 50%の予測を使っている -> 改良が必要。任意に選べる等
    # MPI: MPIの予測値
    # SSP: SSPの予測値
    # wind: MPIの予測値
    # PV: MPIの予測値
    # SoC: 前日の最終SoC obs_listの最後の要素(前episodeの最終timestep)を新しいepisodeでの初期SoCとして使う
    def reset(self):
        state = [
            df_manage.df_input(MIP_q50[1+48*(state_idx-1):48*state_idx]),   # MIP
            df_manage.df_input(SSP_q50[1+48*(state_idx-1):48*state_idx]),   # SSP
            df_manage.df_input(wind_q50[1+48*(state_idx-1):48*state_idx]),  # wind
            df_manage.df_input(PV_q50[1+48*(state_idx-1):48*state_idx]),    # PV
            self.obs_list[-1] # SoC
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

    # 現在の状態と行動に対するrewardを返す(1step分)
    # - rewardは1日(1 episode)ごとに合計される
    # - action > 0 →放電  action < 0 →充電
    # - actionの単位は電力量[kWh or MWh]
    def _get_reward(self, action, SoC):
        ## df.inputからstate_idx(当該time_step)部分のデータを抽出
        # Generation: 発電量
        gen_predict = self.df_input.loc[self.state_idx, "q50"]  # q50: Quantile Regressionによる50%分位点の発電量[MWh]の予測結果。qいくつをとるかは検討の余地あり。
        # gen_observed = self.df_input.loc[self.state_idx,"total_generation_MWh"] # 実績発電量

        # SSP: 電力価格 (single sytem price)
        ssp = self.df_input.loc[self.state_idx, "SSP_q_0.5"] 
        # MIP: 電力価格 (market index price)
        dap = self.df_input.loc[self.state_idx, "DA_price_q_0.5"] 

        # Reward1: Energy Trasnfer（電力系統へ流す売電電力量）を計算する
        # bid_energyはaction + genと0の大きい方を採用する
        # PVの発電量が充電される場合はactionがマイナスになってgen_predictを相殺するので、action + gen_predictとする
        #  action + gen > 0 →action + gen
        #  action + gen < 0 →0
        bid_energy = max(action + gen_predict, 0)
        # rewardを計算
        reward = bid_energy*dap
        # 評価時はこれ：reward = transfer_energy*dap + (transfer_energy-bid_energy)*(ssp-0.07*(transfer_energy-bid_energy))    # 0.07は事前に与えられた固定値（公式document参照）


        # Reward2: 制約条件
        # バッテリー充電が発電出力より高いならペナルティ
        # 越えた量に対してexpのペナルティを与える
        if action < 0 and gen_predict < abs(action): 
            reward += -math.exp(abs(action) - gen_predict)
                
        # Reward3: 制約条件
        # バッテリー放電がSoCより多いならペナルティ
        # 越えた量に対してexpのペナルティを与える
        if SoC < action: 
            reward += -math.exp(action-SoC)

        # Reward4: 制約条件
        # SoCが100％以上でペナルティ
        if self.battery_max_cap < SoC: 
            reward += -math.exp(SoC-self.battery_max_cap)

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
