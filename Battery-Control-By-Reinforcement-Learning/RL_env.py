# インポート：外部モジュール
from cgi import test
from turtle import pos
import gym
import warnings
import numpy as np
import math
import matplotlib.pyplot as plt
from gym import spaces

# internal modules
from RL_dataframe_manager import Dataframe_Manager


warnings.simplefilter('ignore')

class ESS_ModelEnv(gym.Env):
    def __init__(self):
        # データ読込みクラスのインスタンス化
        self.dfmanager = Dataframe_Manager()
        # 学習用のデータ,testデータ、結果格納テーブルを取得
        self.df_train = self.dfmanager.get_train_df()
        self.df_test = self.dfmanager.get_test_df()
        self.df_resultform = self.dfmanager.get_resultform_df()

        # データフレームが正しく読み込まれているか確認
        print(f"Training Data: {self.df_train.head()}")

        # Batteryのパラメーター
        self.battery_max_cap = 4 # 蓄電池の最大容量 ex.4kWh
        self.inverter_max_cap = 4 # インバーターの定格容量 ex.4kW
        self.soc_list = [0.5] # SoC[0,1]の初期値 ex.0.5 (50%)

        ## PPOで使うパラメーターの設定
        # action spaceの定義(上下限値を設定。actionは連続値。)
        # - 1time_step(ex.30 min)での充放電量(規格値[0,1])の上下限値を設定
        # - 本当はinverter_max_capとbattery_max_capを使って、上下限値を設定したい
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # 状態(observation=SoC)の上限と下限の設定
        # observation_spaceの定義(上下限値を設定。observationは連続値。)
        # - 1time_step(ex.30 min)でのSoC(規格値[0,1])の上下限値を設定
        # - PVout, price, imbalance, SoCの4つの値を使っている
        # 1日のステップ数（例：30分間隔で48ステップ）
        self.day_steps = 48

        # 観測空間の次元数を計算（既存の4つ + sin_time + cos_time）
        obs_dim = 4 + 2  # PVout, price, imbalance, SoC, sin_time, cos_time

        # 観測空間の下限と上限を設定
        low = np.array([-np.inf] * obs_dim)
        high = np.array([np.inf] * obs_dim)

        # SoCの範囲を設定（0から1）
        low[3] = 0.0
        high[3] = 1.0

        # sin_timeとcos_timeの範囲を設定（-1から1）
        low[4:6] = -1.0
        high[4:6] = 1.0

        # 観測空間を定義
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # # Rewardの範囲(いらないかも)
        # self.reward_range = (-10000, math.inf) 

        # step関数内でつかうカウンター
        self.state_idx = 0 # time_steps in all episodes (all episodes is a sum of time frames in train/test days 48*days)
        self.reward_total = 0 # 全episodeでの合計のreward
        self.reward_list = [] # 各stepでのreward
        self.episode_rewards = []  # 各エピソードの報酬合計を保存するリスト

           # 特徴量の正規化をここで行います
        # 特徴量の最大値と最小値を取得
        self.pvout_max = self.df_train['PVout'].max()
        self.pvout_min = self.df_train['PVout'].min()
        self.price_max = self.df_train['price'].max()
        self.price_min = self.df_train['price'].min()
        self.imbalance_max = self.df_train['imbalance'].max()
        self.imbalance_min = self.df_train['imbalance'].min()

        # データフレーム内で正規化
        self.df_train['PVout'] = (self.df_train['PVout'] - self.pvout_min) / (self.pvout_max - self.pvout_min)
        self.df_train['price'] = (self.df_train['price'] - self.price_min) / (self.price_max - self.price_min)
        self.df_train['imbalance'] = (self.df_train['imbalance'] - self.imbalance_min) / (self.imbalance_max - self.imbalance_min)

        # # Mode選択
        # self.mode = train    # train or test

    #### time_stepごとのactionの決定とrewardの計算を行う
    # - trainのときに使う。testのときはstep_for_testを使う
    def step(self, action):
        # 現在のステップ数を表示
        #print(f"Current step (state_idx): {self.state_idx}")
        
        # actionの値を表示
        #print(f"Step called with action: {action}")

         #actionをリストに保存し、csv形式で保存
        with open("Battery-Control-By-Reinforcement-Learning/RL_action.csv", "a") as f:
            f.write(str(action[0]) + "\n")

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
        # actionのスケーリング [kWh]（-1.0〜1.0 の範囲を -inverter_max_cap〜inverter_max_cap に変換）
        action_value = action * self.inverter_max_cap * 0.5  # スケール調整

        # 新しいSoCを計算
        new_soc = self.soc_list[-1] - (action_value / self.battery_max_cap)  # SoCは0〜1の範囲
        #new_soc = np.clip(new_soc, 0, 1) # SoCが0~1の範囲内に収まるようにするか、ペナルティを与えるか
         # 各timestepでのSoCをobsをリストに追加
        self.soc_list.append(new_soc)


        reward = self._get_reward(action_value, self.soc_list[-1])              
        self.reward_list.append(reward)
        #print(f"Reward list: {self.reward_list}")
        # 全episodeでのrewardを計算
        self.reward_total += self.reward_list[-1]

            # 時間情報の計算
        day_steps = 48
        time_of_day = self.state_idx % day_steps
        theta = 2 * np.pi * time_of_day / day_steps
        sin_time = np.sin(theta)
        cos_time = np.cos(theta)


       

        # soc_listの最後の要素(前timestepのSoC)にactionを足す
        # actionをnp.float32型からfloat型に変換してから足す。stable_baselines3のobservatuionの仕様のため。
        observation = [
            self.df_train["PVout"][self.state_idx], # PV発電量実績値
            self.df_train["price"][self.state_idx], # 電力価格実績値
            self.df_train["imbalance"][self.state_idx], # インバランス価格実績値
            new_soc, # 新しいSoC
            sin_time,
            cos_time
        ]

        #print(f"Observation: {observation}")

       

        # checking whether our episode (day) ends
        # - 1日(1 episode)が終わったら、done = Trueにする
        # state_idxは48コマ(1日)で割った余りが0になると、1日終了とする
        if self.state_idx % 48 == 0:
            done = True # Trueだと勝手にresetが走る
            # 直近48コマの報酬の合計を計算しリストに追加
            recent_reward = sum(self.reward_list[-48:])
            self.episode_rewards.append(recent_reward)
            info = {'episode_reward': recent_reward}  # 情報にエピソードの合計報酬を追加
        else:
            done = False    
             # 付随情報をinfoに入れる
            info = {}          
        
        return observation, reward, done, info
    
    
    ## def reset(self)
    ## 状態の初期化: trainingで1episode(1日)終わると呼ばれる
    # - PPOのlearn(RL_train.py内にある)を走らせるとまず呼ばれる。
    # - RL_env.pyのstepメソッドにおいて、done = Trueになると呼ばれる。doneを制御することで任意のタイミングでresetを呼ぶことができる。
    # - 1episode(1日)終わるとresetを呼ぶように実装してある(def step参照)ので、次の日のデータ(48コマ全て)をstateへ入れる
    # - 現状ではdeterministicな予測を使っている -> 改良が必要。確率的になるべき。
    # PVout: PV発電量の実績値
    # price: 電力価格の実績値
    # imablance: インバランス価格の実績値
    # SoC: 前日の最終SoC soc_listの最後の要素(前episodeの最終timestep)を新しいepisodeでの初期SoCとして使う
    def reset(self):
        # データセット内の総日数を計算
        total_days = len(self.df_train) // self.day_steps - 1

        # ランダムに日付を選択
        random_day = np.random.randint(0, total_days)

        # state_idxをランダムな日の開始インデックスに設定
        self.state_idx = random_day * self.day_steps

        # 報酬やその他の変数をリセット
        self.reward_total = 0
        self.reward_list = []

        # オプションで、ランダムな初期SoCから始める
        initial_soc = np.random.uniform(0, 1)
        self.soc_list = [initial_soc]

        # 時間情報の計算
        time_of_day = self.state_idx % self.day_steps
        theta = 2 * np.pi * time_of_day / self.day_steps
        sin_time = np.sin(theta)
        cos_time = np.cos(theta)

        # 初期状態の観測値を取得
        observation = [
            self.df_train["PVout"][self.state_idx],
            self.df_train["price"][self.state_idx],
            self.df_train["imbalance"][self.state_idx],
            initial_soc,
            sin_time,
            cos_time
        ]
        return observation

    ## _get_reward(self, action, SoC, trainOrTest)
    ## 現在の状態と行動に対するrewardを返す(1step分)
    # - rewardは1日(1 episode)ごとに合計される
    # - rewardは学習(train)の場合でしか使わない（testでは使わない）
    # - action > 0 →放電  action < 0 →充電
    # - actionの単位は電力量[kWh or MWh]
    def _get_reward(self, action, SoC):
        ## df.trainからstate_idx(当該time_step)部分のデータを抽出
        # Generation: PV発電量(正規化済み)
        pv_gen_normalized= self.df_train.loc[self.state_idx, "PVout"]  # PV発電実績値（正規化済み）
         # 元のスケールに戻す
        pv_gen = pv_gen_normalized * (self.pvout_max - self.pvout_min) + self.pvout_min

        # 電力価格（正規化済み）
        price_normalized = self.df_train.loc[self.state_idx, "price"]  # 電力価格実績値（正規化済み）
        price = price_normalized * (self.price_max - self.price_min) + self.price_min

        # **インバランス価格を取得して非正規化**
        imbalance_price_normalized = self.df_train.loc[self.state_idx, "imbalance"]
        imbalance_price = imbalance_price_normalized * (self.imbalance_max - self.imbalance_min) + self.imbalance_min

        # Reward1: Energy Trasnfer（電力系統へ流す売電電力量）を計算する
        # bid_energyはaction + genと0の大きい方を採用する
        # PVの発電量が充電される場合はactionがマイナスになってpv_genを相殺するので、action + pv_genとする
        #  action + gen > 0 →action + gen
        #  action + gen < 0 →0
        #  action,pv_gen,bid_energyは[kWh]であることに注意

         # **予測誤差を考慮する**
        # 予測誤差をシミュレーション（平均0、標準偏差sigmaの正規分布からサンプリング）
        sigma = 0.1  # 予測誤差の標準偏差を適切に設定
        forecast_error = np.random.normal(0, sigma)

        # PV発電量の予測値を計算
        pv_gen_forecast = pv_gen + forecast_error

        # 予定していたエネルギー供給量（入札量）を計算
        scheduled_energy = max(action + pv_gen_forecast, 0)

        # 実際のエネルギー供給量を計算
        actual_energy_delivered = max(action + pv_gen, 0)

        # インバランスエネルギーの計算（絶対値を取る）
        imbalance_energy = abs(scheduled_energy - actual_energy_delivered)

        # Reward1: Energy Transfer（電力系統へ流す売電電力量）を計算する
        # 実際の売電電力量を使用
        bid_energy = actual_energy_delivered
        positive_reward = bid_energy * price

        # **インバランスコストを計算**(インバランスエネルギー×インバランス価格)
        imbalance_cost = imbalance_energy * imbalance_price

       # 制約条件のペナルティを追加
        penalty = 0
        penalty_weight = 100  # ペナルティの重み
        # Reward2: バッテリー充電がPV発電量を超える場合のペナルティ
        if action < 0 and abs(action) > pv_gen:
            penalty -= penalty_weight * (abs(action) - pv_gen)  # 超過量に比例したペナルティ
        # Reward3: バッテリー放電がSoCを超える場合のペナルティ
        if action > 0 and action > SoC:
            penalty -= penalty_weight * (action - SoC)  # 超過量に比例したペナルティ  
        # Reward4: SoCが0〜1の範囲外になる場合のペナルティ
        if SoC < 0:
            penalty -= penalty_weight * abs(SoC)  # SoCが0未満の場合のペナルティ
        elif SoC > 1:
            penalty -= penalty_weight * (SoC - 4)  # SoCが1を超えた場合のペナルティ

        # **インバランスコストを考慮した最終報酬**
        reward = positive_reward - imbalance_cost + penalty
        
        

        return reward


    ## 報酬を決定する前に、手動で実現可能なactionへ修正する場合はこのメソッドを使う（改変中）
    # def _get_possible_schedule(self, action):
    #     # 入力データの設定
    #     self.PV_out_time = self.PVout[self.time]
    #     self.price_time = self.price[self.time]
    #     self.imbalance_time = self.imbalance[self.time]
                
    #     #時刻self.timeに対応するデータを取得
    #     self.input_price = self.price[48*(self.days - 1) + self.time]
    #     self.input_PV = self.PVout[48*(self.days - 1) + self.time]

    #     #### actionを適正化(充電をPVの出力があるときのみに変更)
    #     # PV発電量が0未満の場合、0に設定
    #     if self.PV_out_time < 0:
    #         self.PV_out_time = [0]
    #     # 充電時、PV発電量<充電量 の場合、充電量をPV出力値へ調整
    #     if self.PV_out_time < -action and action < 0:
    #         action_real = -self.PV_out_time
    #     # 放電時、放電量>蓄電池残量の場合、放電量を蓄電池残量へ調整
    #     elif action > 0 and 0 < self.battery < action:
    #         action_real = self.battery
    #     # 充電時、蓄電池残量が定格容量に達している場合、充電量を0へ調整
    #     elif self.battery == self.battery_MAX and action < 0:
    #         action_real = 0
    #     # 放電時、蓄電池残量が0の場合、放電量を0へ調整
    #     elif action > 0 and self.battery == 0:
    #         action_real = 0
    #     # 上記条件に当てはまらない場合、充放電量の調整は行わない
    #     else:
    #         action_real = action
    #     # 実際の充放電量をリストに追加
    #     self.all_action_real.append(action_real)

    #     #### 蓄電池残量の更新
    #     # 次のtimeにおける蓄電池残量を計算
    #     next_battery = self.battery - action_real*0.5 #action_real*0.5とすることで[kWh]へ変換

    #     ### 蓄電池残量の挙動チェック
    #     # 次のtimeにおける蓄電池残量が定格容量を超える場合、定格容量に制限
    #     if next_battery > self.battery_MAX:
    #         next_battery = self.battery_MAX
    #     # 次のtimeにおける蓄電池残量が0kwh未満の場合、0に制限
    #     elif next_battery < 0:
    #         next_battery = 0
    #     # 充電の場合、PV発電量から充電量を差し引く
    #     if action_real < 0:
    #         self.PV_out_time = self.PV_out_time + action_real