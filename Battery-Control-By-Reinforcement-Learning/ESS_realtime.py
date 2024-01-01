import gym
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import math as ma
import tkinter as tk
#import tensorflow as tf

from matplotlib.backends.backend_pdf import PdfPages
from stable_baselines3 import PPO
#from torch.utils.tensorboard import SummaryWriter # tensorBoardを起動して、学習状況を確認する

print("\n---充放電計画策定プログラム(リアルタイム)開始---\n")

warnings.simplefilter('ignore')



class ESS_model(gym.Env):
    def __init__(self, mode, pdf_day, train_days, test_day, PV_parameter, action_space):
        #パラメータの定義
        self.episode = 0
        self.end_count = 0  # 追加エピソード
        self.total_step = action_space # 1Dayの総コマ数
        self.gamma = ma.exp(-(1/action_space)) # 放電に対する割引率
        self.omega = ma.exp(1/action_space) # 充電に対する割引率
        self.battery_MAX = 4 # 蓄電池の定格容量4kWh
        self.MAX_reward = -10000
        self.Train_Days = train_days # 学習Day
        self.test_days = test_day - 1 # テストDay数
        
        # mode select
        self.mode = mode
        if mode == "train":
            self.last_day = self.Train_Days
        elif mode == "test":
            self.last_day = self.test_days
        self.all_rewards = []

        # データのロード
        print("-データロード-")
        # 学習データ
        input_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
        # テストデータ(これが充放電計画策定したいもの)
        predict_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv")

        # 学習データの日数+1日分データが必要
        # 空データドッキング
        data = [[0] * 20] * 48
        columns = ["year","month","day","hour","temperature","total precipitation","u-component of wind","v-component of wind","radiation flux","pressure","relative humidity","PVout","price","imbalance",
                   "yearSin","yearCos","monthSin","monthCos","hourSin","hourCos"]
        new_rows_df = pd.DataFrame(data, columns=columns)
        input_data = pd.concat([input_data, new_rows_df], ignore_index=True)

        # データの作成
        print("データ作成")
        if self.mode == "train":
            # 30分単位のため、料金を0.5倍
            price = input_data["price"]/2   # [JPY/kWh] -> [JPY/kW/30min]
            imbalance = input_data["imbalance"]/2   # [JPY/kWh] -> [JPY/kW/30min]

            PVout = input_data["PVout"] # [kW]

            # 値を格納
            price_data = price
            imbalance_data = imbalance
            PVout_data = PVout
            

        elif self.mode == "test":
            # 30分単位のため、料金を0.5倍
            price = predict_data["price"]/2   # [JPY/kWh] -> [JPY/kW/30min]
            imbalance = predict_data["imbalance"]/2   # [JPY/kWh] -> [JPY/kW/30min]
            #self.PV = PV_parameter #upper, lower, PVoutの選択用、現在使ってないが、今後のために保留
            PVout = predict_data["PVout"]   # [kW]
            
            # 値を格納
            price_data = price
            imbalance_data = imbalance
            PVout_data = PVout
           

        # pandas -> numpy変換,型変換
        print("データ変換")
        self.price_all = price_data.values
        self.price = self.price_all.reshape((len(self.price_all), 1)) 

        self.imbalance_all = imbalance_data.values
        self.imbalance = self.imbalance_all.reshape((len(self.imbalance_all), 1)) 

        self.PVout_all = PVout_data.values
        self.PVout = self.PVout_all.reshape((len(self.PVout_all), 1))

        # アクション
        self.ACTION_NUM = action_space #アクションの数(現状は48の約数のみ)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape = (self.ACTION_NUM,))

        # 状態の上限と下限の設定
        low_box = np.zeros(self.ACTION_NUM*2+1) # 入力データの下限値×入力データの数
        high_box = np.ones(self.ACTION_NUM*2+1) # 入力データの上限値×入力データの数
        LOW = np.array(low_box)
        HIGH = np.array(high_box)
        self.observation_space  = gym.spaces.Box(low=LOW, high=HIGH)

        # 初期データの設定
        self.reset()


    #### timeごとのrewardの計算
    def step(self, action): 
        done = False # True:終了　False:学習継続

        #初期のreward
        reward = 0
        all_action = action #[kW]

        #action > 0 →放電  action < 0 →充電
        for self.time_stamp in range(0, self.ACTION_NUM):

            # float型へ
            action = float(all_action[self.time_stamp])

            ACTION = action*1.2 # [kW]  #値を拡大することでrewardへ影響力を持たせる
            ACTION = round(ACTION, 1) # 小数点第1位にまるめる
            time = self.time
            count = self.count
            soc = (self.battery / self.battery_MAX) # %

            self.all_soc.append(soc*100)
            self.all_battery.append(self.battery)
            self.all_price.append(self.price_time)
            self.all_time.append(time/2)
            self.all_count.append(count/2)
            self.all_action.append(ACTION)
            self.all_PV_out_time.append(self.PV_out_time[0])
            self.all_imbalance.append(self.imbalance)


            #### actionを適正化(充電をPVの出力があるときのみに変更)
            # PV発電量が0未満の場合、0に設定
            if self.PV_out_time < 0:
                self.PV_out_time = [0]
            # 充電時、PV発電量<充電量で場合、充電量をPV出力値へ調整
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

            #### rewardの計算
            # 評価用の充電残量
            n_battery = self.battery - ACTION*0.5 #action_real*0.5とすることで[kWh]へ変換
            # これまでのrewardに時刻self.timeのrewardを加算
            reward += self.reward_set(ACTION ,n_battery)

            #print(self.time_stamp, reward)

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
            # 放電量のみ抽出
            if action_real > 0:
                temp = float(action_real)
            else:
                temp = 0
            
            # 発電量(充電量のぞく)+放電量
            energy_transfer = self.PV_out_time[0] + temp #[kW]
            
            #print(type(energy_transfer), type(self.PV_out_time[0]), type(temp))
            self.all_energy_transfer.append(energy_transfer)

            # 入力データ(学習時：実測　テスト時：予測)
            self.data_set()
    
        # 現在のrewardをself.rewardsリストに追加
        self.rewards.append(reward)

        # timeが最終かつ学習データ最終日(エピソード終了時)に記録
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
            if self.end_count >= 20000:
                if self.episode == 100000 or self.episode > 20000:
                    self.evalution("Battery-Control-By-Reinforcement-Learning/" + "result-" + self.mode + "-end.pdf")
                    self.model.save("ESS_model_end")
                    #done = True # 学習終了
                    self.end_count = 0

            # エピソード数表示
            # print("episode:"+str(self.episode) + "/" + str(episode + self.end_count))
            print("episode:"+str(self.episode) + "/" + str(episode))

        # testモードの最後に結果をpdfで保存
        if time == 48 and self.days == self.last_day and self.mode == "test":
            self.evalution("Battery-Control-By-Reinforcement-Learning/" + "result-" + self.mode + ".pdf")
        # エピソードが終了の場合は状態をリセット
        if time == 48 and self.days == self.last_day:
            state = self.reset()
        # 新たな状態を生成
        else:
            state = [soc]
            state.extend(self.input_PV_data)
            state.extend(self.input_price_data)

        return state, reward, done, {}
    
    # 状態の初期化
    def reset(self):
        # 結果データ
        result_dataframe = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        # 蓄電池データ取得
        # dataframe最終行から取得
        last_soc_actual_row = result_dataframe[result_dataframe['SoC_actual_realtime'].notna()].tail(1)

        # エラー処理：SoCのデータがdaraframeになければ0に設定する
        if not last_soc_actual_row.empty:
            last_soc_actual_value = last_soc_actual_row['SoC_actual_realtime'].values[0]
            now_battery = last_soc_actual_value * 0.01 * self.battery_MAX  # 電力量[kWh]に変換
        else:
            now_battery = 0

        self.time = 0
        self.count = 0
        self.battery = now_battery    #max:4[kWh]
        self.days = 1
        self.rewards = []
        self.all_PV_out_time = []
        self.all_soc = []
        self.all_battery = []
        self.all_price = []
        self.all_time = []
        self.all_count = []
        self.all_action = []    # 蓄電池動作(修正なし)
        self.all_action_real = []   # 蓄電池動作(修正あり)
        self.all_imbalance = []
        self.all_energy_transfer = []

        self.data_set()
        state = [self.battery]
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

    # timeごとのrewardの計算 > rewardの設定内容
    def reward_set(self, ACTION, n_battery):
        #ACTION > 0 →放電  ACTION < 0 →充電
        reward = 0

        # 現在の状態と行動に対するreward
        # rewardはすべて入出力[kW]*値段[JPY/30min]で計算(実際の報酬ではない)
        # 充電する場合
        if ACTION <= 0:
            # 売電(PV出力-BT入力)に対するreward(今の状態×行動)
            if -ACTION < self.input_PV:
                reward += ((self.omega)**(self.time_stamp))*self.input_price*(self.PV_out_time + ACTION)*0
            # BT入力がPV出力より高いならペナルティ(今の状態×行動)
            if -ACTION > self.input_PV:
                reward += ((self.omega)**(self.time_stamp))*self.input_price*ACTION
        
        # 放電する場合
        if ACTION > 0:
            # PV出力(売電)に対するreward
            reward += ((self.gamma)**(self.time_stamp))*self.input_price*self.PV_out_time*0
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

    #### timeごとのrewardの計算 > グラフ作成
    def evalution(self, pdf_name):
        pp = PdfPages(pdf_name) # PDFの作成
        if self.mode == "train":
            graph_1 = self.graph(self.all_rewards)
            pp.savefig(graph_1)
        graph_2 = self.schedule(self.all_action_real, self.all_PV_out_time, self.all_energy_transfer, self.all_soc, mode = 0)
        graph_3 = self.schedule(self.all_action_real, self.all_PV_out_time, self.all_energy_transfer, self.all_soc, mode = 1)

        pp.savefig(graph_2)
        pp.savefig(graph_3)

        pp.close()

    #### timeごとのrewardの計算 > グラフ作成 > 充放電計画のデータ出力(名前変える)
    def schedule(self, action, PVout, energy_transfer, soc, mode):     #修正後のactionを引き渡す
        ## test時のtime_stampを取得
        #入力データから取得
        predict_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv")
        year_stamp = predict_data["year"]
        month_stamp = predict_data["month"]
        day_stamp = predict_data["day"]
        hour_stamp = predict_data["hour"]

        # hour_stampを整数化
        hour_stamp_ = [int(hour) for hour in hour_stamp]
        # minute_stampをhourの小数点第一位に応じて設定
        minute_stamp = [0 if int(hour * 10) % 10 == 0 else 30 for hour in hour_stamp]
        
        # 時系列として統合
        time_stamp = pd.to_datetime({'year': year_stamp, 'month': month_stamp, 'day': day_stamp, 'hour': hour_stamp_, 'minute': minute_stamp})


        fig = plt.figure(figsize=(22, 12), dpi=80)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax2.set_ylim([-1, 101])
        ax1.tick_params(axis='x', labelsize=35)
        ax1.tick_params(axis='y', labelsize=35)
        ax2.tick_params(axis='x', labelsize=35)
        ax2.tick_params(axis='y', labelsize=35)
        
        if self.mode == "train":
            # プロット
            ax1.plot(self.all_time, action, "blue", drawstyle="steps-post", label="Charge and discharge")
            ax1.plot(self.all_time, PVout, "Magenta", label="PV generation")
            ax2.plot(self.all_time, soc, "red", label="SoC")
            # 横軸の目盛りを設定
            ax1.set_xticks(np.arange(0, 24, 6))
            ax2.set_xticks(np.arange(0, 24, 6))
            
        elif self.mode == "test":
            # プロット
            ax1.plot(time_stamp, action, "blue", drawstyle="steps-post",label="Charge and discharge")
            ax1.plot(time_stamp, PVout, "Magenta",label="PV generation")
            ax2.plot(time_stamp, soc, "red",label="SoC")
            # 横軸の設定
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))  # 時刻のフォーマット
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # 6時間ごとに目盛りを設定
            plt.xticks(rotation=45)  # x軸のラベルを回転


        if mode == 0: # 電力価格ありのグラフ
            if self.mode == "train":
                ax1.plot(self.all_time, self.all_price, "green", drawstyle="steps-post", label="Power rates")
            elif self.mode == "test":
                ax1.plot(time_stamp, self.all_price, "green", drawstyle="steps-post", label="Power rates")
            ax1.set_ylabel("Power [kW] / Power rates [JPY/30min]", fontsize=35)
        elif mode == 1:
            ax1.set_ylim([-2, 2])
            ax1.set_ylabel("Power [kW]", fontsize=35)    
        
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', prop={"size": 35}).get_frame().set_alpha(0.0)

        if self.mode == "train":
            ax1.set_xlim([0, 23.5])
        elif self.mode == "test":
            # 1日分を想定した設定(0～47)
            ax1.set_xlim([time_stamp[0], time_stamp[47]])
        ax1.set_xlabel('Time [hour]', fontsize=35)
        ax1.grid(True)
        ax2.set_ylabel("SoC[%]", fontsize=35)
        plt.tick_params(labelsize=35)
        plt.close()

        if self.mode == "test":

            # テストデータの時刻
            year_stamp = pd.DataFrame(year_stamp)
            month_stamp = pd.DataFrame(month_stamp)
            day_stamp = pd.DataFrame(day_stamp)
            hour_stamp = pd.DataFrame(hour_stamp)

            # 値の形式変換
            action = [x.item() if isinstance(x, np.ndarray) else x for x in action]
            soc = [x.item() if isinstance(x, np.ndarray) else x for x in soc]

            action = pd.DataFrame(action)
            PVout = pd.DataFrame(PVout)
            soc = pd.DataFrame(soc) 
            energytransfer = pd.DataFrame(energy_transfer)
            price = pd.DataFrame(self.price)
            imbalance = pd.DataFrame(self.imbalance)

            # データ結合
            result_data = pd.concat([year_stamp,month_stamp],axis=1)
            result_data = pd.concat([result_data,day_stamp],axis=1)
            result_data = pd.concat([result_data,hour_stamp],axis=1)
            result_data = pd.concat([result_data,action],axis=1)
            result_data = pd.concat([result_data,PVout],axis=1)
            result_data = pd.concat([result_data,soc],axis=1)
            result_data = pd.concat([result_data,energytransfer],axis=1)
            result_data = pd.concat([result_data,price],axis=1)
            result_data = pd.concat([result_data,imbalance],axis=1)

            label_name = ["year","month","day","hour","charge/discharge","PVout","SoC","energy_transfer","price","imbalance"] # 列名
            result_data.columns = label_name # 列名付与
            result_data.to_csv("Battery-Control-By-Reinforcement-Learning/result_data.csv")

        return fig

    # timeごとのrewardの計算 > グラフ作成 > rewardの推移のグラフ出力
    def graph(self, y):
        fig = plt.figure(figsize=(24, 14), dpi=80)
        plt.plot(np.arange(self.episode), y, label = "Reward")
        plt.legend(prop={"size": 35})
        plt.xlabel("Episode", fontsize = 35)
        plt.ylabel("Reward", fontsize = 35)
        plt.tick_params(labelsize=35)
        plt.close()
        
        return fig

    # メインルーチン   
    def main_root(self, mode, num_episodes, train_days, episode, model_name):
        
        # #Tkinter処理 epsode途中に終了を防ぐ
        #root = tk.Tk()
        #root.withdraw()
        
        if mode == "train":
            print("モデル学習開始")
            self.model = PPO("MlpPolicy", env, gamma = 0.8, gae_lambda = 1, clip_range = 0.2, 
                            ent_coef = 0.005, vf_coef =0.5, learning_rate = 0.0001, n_steps = 48, 
                            verbose=1, tensorboard_log="./PPO_tensorboard/") 
            #モデルの学習
            self.model.learn(total_timesteps=num_episodes*train_days*episode)
            print("モデル学習終了")

        
        if mode == "test":
            #モデルのロード
            print("モデルロード")
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
episode = 5 # 10000000  

print("-Trainモード開始-")

# test 1Day　Reward最大
pdf_day = 0 #確率密度関数作成用のDay数 75 80
train_days = 366 # 学習Day数 70 ~ 73
test_day = 3 # テストDay数 + 2 (最大89)
PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類)　#今後はUpper, lower, PVout
mode = "train" # train or test
model_name = "ESS_model" # ESS_model ESS_model_end

# Training環境設定と実行
#env = ESS_model(mode, pdf_day, train_days, test_day, PV_parameter, action_space)
#env.main_root(mode, num_episodes, train_days, episode, model_name)# Trainingを実行

print("-Trainモード終了-")

print("-充放電計画策定開始-")

# test 1Day　Reward最大
pdf_day = 0 #確率密度関数作成用のDay数 75 80
train_days = 366 # 学習Day数 70 ~ 73
test_day = 3 # テストDay数 + 2 (最大89)
PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類) #今後はUpper, lower, PVout
mode = "test" # train or test
model_name = "ESS_model" # ESS_model ESS_model_end

# Test環境設定と実行 学習
env = ESS_model(mode, pdf_day, train_days, test_day, PV_parameter, action_space)
env.main_root(mode, num_episodes, train_days, episode, model_name)

print("-充放電計画策定終了-")


print("\n---充放電計画策定プログラム(リアルタイム)終了---\n")
