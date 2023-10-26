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

warnings.simplefilter('ignore')

class ESS_ModelEnv(gym.Env):
    def __init__(self, mode, train_days, test_day):      
        # PPOで使うパラメーターの設定        
        self.gamma = math.exp(-(1/action_space)) # 放電に対する割引率
        self.omega = math.exp(1/action_space) # 充電に対する割引率
        self.battery_max_cap = 4 # 蓄電池の定格容量4kWh
        self.reward_range = (-10000, math.inf)
        self.train_Days = train_days # 学習Day
        self.test_days = test_day - 1 # テストDay数
        self.timesteps = 100 # 1つのtime_frameの中で何度action -> observe -> rewardを繰り返すかの設定値
        self.battery

        # Action spaceの定義(上下限値を設定。actionは連続値。)
        self.action_spcae = gym.spaces.Box(low=-1.0, high=1.0) 

        # 状態の上限と下限の設定
        self.observation_space  = gym.spaces.Box(low=0, high=1)


    #### timeごとのrewardの計算
    def step(self, action):
        done = False # True:終了　False:学習継続

        #action > 0 →放電  action < 0 →充電
        for self.time_stamp in range(0, self.timesteps):
            soc = (self.battery_cunnret_energy / self.battery_max_cap) # SoC∈[0,1]へ変換
            #### rewardの計算
            # 評価用の充電残量
            n_battery = self.battery - action
            # これまでのrewardに時刻self.timeのrewardを加算
            reward += self.reward_set(ACTION ,n_battery)

            # SoC算出
            self.battery = next_battery
            soc = (self.battery / self.battery_MAX) # %

            ## timeデータ更新         
            self.time += 1
            time = self.time
            self.count += 1

            # timeが最終コマのとき
            if self.time == 48:
                self.days += 1
                self.time = 0

            # 売電量の更新
            energy_transfer = self.PV_out_time[0] * 0.5 #[kW]->[kWh]
            self.all_energy_transfer.append(energy_transfer)

            # 入力データ(学習時：実測　テスト時：予測)
            self.data_set()
    
        # 現在のrewardをself.rewardsリストに追加
        # action_space(ex. 12コマ分)の合計の報酬を記録
        self.rewards.append(reward)

        # timeが最終かつ学習データ最終日(エピソード終了時)に記録
        # [2023.09.07 小平]　最終日とはどういう意味？日はepisodeで対応しているのではないか？
        if time == 48 and self.days == self.last_day and self.mode == "train": #学習の経過表示、リセット
            # 最初のエピソードのときに値を設定
            if self.episode == 0:
                self.MAX_reward = np.sum(self.rewards)

            # エピソード数更新
            self.episode += 1
            #print("episode:"+str(self.episode) + "/"+str(episode) + " + " + str(self.end_count))
            # 現在のエピソードのall_rewardsをself.all_rewardsリストに追加
            self.all_rewards.append(np.sum(self.rewards))

            # モデルの報酬の最高値を更新した場合はモデル"ESS_model"を保存
            if np.sum(self.rewards) >= self.MAX_reward:
                self.MAX_reward = np.sum(self.rewards) # rewardの最高値
                self.evalution("Battery-Control-By-Reinforcement-Learning/" + "result-" + self.mode + ".pdf")
                self.model.save("ESS_model")
                self.end_count = 0
            # モデルの報酬の最高値を更新できなかった場合はend_count(追加エピソード)を設定
            # 動作詳細不明
            elif np.sum(self.rewards) < self.MAX_reward:
                self.end_count += 1

            # end_countが20000以上になった場合は十分学習したと判定し、モデル"ESS_model_end"を保存し終了
            # [2023.09.05 小平]　end_countとepisodeの違いがよくわからない
            if self.end_count >= 20000:
                if self.episode == 100000 or self.episode > 20000:
                    self.evalution("Battery-Control-By-Reinforcement-Learning/" + "result-" + self.mode + "-end.pdf")
                    self.model.save("ESS_model_end")
                    done = True # 学習終了　＃ [2023.09.05 小平]　なぜコメントアウトされている？
                    self.end_count = 0

            # エピソード数表示
            # print("episode:"+str(self.episode) + "/" + str(episode + self.end_count))  # [2023.09.05 小平]　これいる？
            print("episode:"+str(self.episode) + "/" + str(episode))

        # testモードの最後に結果をpdfで保存
        if time == 48 and self.days == self.last_day and self.mode == "test":
            self.evalution("Battery-Control-By-Reinforcement-Learning/" + "result-" + self.mode + ".pdf")
        # エピソードが終了の場合は状態をリセット
        # [2023.09.05 小平]　なぜリセットする必要があるのか？ -> episodeごとに学習状況をリセットする必要があるため。
        if time == 48 and self.days == self.last_day:
            state = self.reset()
        # 新たな状態を生成
        else:
            state = [soc]
            state.extend(self.input_PV_data)
            state.extend(self.input_price_data)
        return state, reward, done, {}
    
    # 状態の初期化：[2023.09.05 小平] 各要素の説明がほしい
    def reset(self):
        self.time = 0
        self.count = 0
        self.battery = 0
        self.days = 1
        self.rewards = []
        self.all_PV_out_time = []
        self.all_soc = []
        self.all_battery = []
        self.all_price = []
        self.all_time = []
        self.all_count = []
        self.all_action = []    # 蓄電池動作(修正なし)　[2023.09.05 小平]　具体的に何を修正？
        self.all_action_real = []   # 蓄電池動作(修正あり)　[2023.09.05 小平]　具体的に何を修正？
        self.all_imbalance = []
        self.all_energy_transfer = []

        self.data_set()
        # [2023.09.05 小平] stateの要素に何が入る（べきな）のかを説明する
        state = [self.battery/4] # [2023.09.05 小平]　これは初期SoCを設定している。根拠は特に無く25%(1/4)に設定。
        state.extend(self.input_PV_data)
        state.extend(self.input_price_data)

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
    def calc_reward(self, ACTION, n_battery):
        #ACTION > 0 →放電  ACTION < 0 →充電
        reward = 0

        # 現在の状態と行動に対するreward
        # rewardはすべて入出力[kW]*値段[JPY/30min]で計算(実際の報酬ではない)
        # 充電する場合
        if ACTION <= 0:
            # 売電(PV出力-BT入力)に対するreward(今の状態×行動)
            if -ACTION < self.input_PV:
                reward += ((self.omega)**(self.time_stamp))*self.input_price*(self.PV_out_time + ACTION)
            # BT入力がPV出力より高いならペナルティ(今の状態×行動)
            if -ACTION > self.input_PV:
                reward += ((self.omega)**(self.time_stamp))*self.input_price*ACTION
        
        # 放電する場合
        if ACTION > 0:
            # PV出力(売電)に対するreward
            reward += ((self.gamma)**(self.time_stamp))*self.input_price*self.PV_out_time
            # BT出力がSoCより大きいならペナルティ(今の状態×行動)
            if ACTION > self.battery: 
                reward += ((self.omega)**(self.time_stamp))*self.input_price*(self.battery - ACTION)
            # BT出力(売電)に対するreward(今の状態×行動)...PV出力に加えてBT出力が報酬として加算される
            if ACTION <= self.battery:
                reward += ((self.gamma)**(self.time_stamp))*self.input_price*ACTION

        # 次の状態と行動に対するreward
        # SoCが100％以上でペナルティ
        if n_battery > self.battery_MAX: 
            reward += ((self.omega)**(self.time_stamp))*self.input_price*(-n_battery)

        return reward

    def _get_possible_schedule(action):
        #### actionを適正化(充電をPVの出力があるときのみに変更)
        # PV発電量が0未満の場合、0に設定
        if self.PV_out_time < 0:
            self.PV_out_time = [0]
        # 充電時、PV発電量<充電量 の場合、充電量をPV出力値へ調整
        if self.PV_out_time < -ACTION and action < 0:
            action_real = -self.PV_out_time
        # 放電時、放電量>蓄電池残量の場合、放電量を蓄電池残量へ調整
        elif action > 0 and 0 < self.battery < ACTION:
            action_real = self.battery
        # 充電時、蓄電池残量が定格容量に達している場合、充電量を0へ調整
        elif self.battery == self.battery_MAX and action < 0:
            action_real = 0
        # 放電時、蓄電池残量が0の場合、放電量を0へ調整
        elif action > 0 and self.battery == 0:
            action_real = 0
        # 上記条件に当てはまらない場合、充放電量の調整は行わない
        else:
            action_real = ACTION
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
