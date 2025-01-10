# インポート：外部モジュール
from cgi import test
import time
from turtle import pos
import gym
import warnings
import numpy as np
import math
import matplotlib.pyplot as plt
from gym import spaces
from torch import cos_
import pandas as pd
import math
from scipy.stats import truncnorm

# internal modules
from RL_dataframe_manager import Dataframe_Manager
from RL_operate import Battery_operate as Operate


warnings.simplefilter('ignore')

class ESS_ModelEnv(gym.Env):
    def __init__(self, mode):
        ## for debug
        self.end_terminated_state = []
        self.end_truncated_state = []
        self.episode_count = 0
        # RL_action.csvを各エピソードごとにcolumnを作って保存したい(RL_train.pyでやるかここでやるかは要検討)
        
        # データ読込みクラスのインスタンス化
        self.dfmanager = Dataframe_Manager()
        # 学習用のデータ,testデータ、結果格納テーブルを取得
        self.df_train = self.dfmanager.get_train_df()
        # 何に使うのか不明
        if mode == "bid":
            self.df_test = self.dfmanager.get_test_df_bid()
        elif mode == "realtime":
            self.df_test = self.dfmanager.get_test_df_realtime()
        # self.df_resultform = self.dfmanager.get_resultform_df()
        # データフレームが正しく読み込まれているか確認
        print(f"Training Data: {self.df_train.head()}")

        # Batteryのパラメーター
        self.battery_max_cap = 4.0 # 蓄電池の最大容量 ex.4kWh
        self.inverter_max_cap = 4.0 # インバーターの定格容量 ex.4kW

        # 1日のステップ数（例：30分間隔で48ステップ）
        self.day_steps = 48

        ## reset()で初期化---------------------------------------------------------
        self.soc_list = [] 
        self.current_episode_reward = 0 # 現在のエピソードの合計報酬
        self.reward_list = [] # 現在のエピソードでの各stepのreward
        self.state_idx = 0 # time step
        self.current_imbalance = 0 # エピソードの合計インバランス
        self.action_difference_ = 0 # RLからのアクションと実際のアクションの差分
        self.current_deal_profit = 0 # エピソードの合計取引利益

        # debug
        self.current_action_sum = 0
        ## ------------------------------------------------------------------------
        self.episode_rewards_summary = []  # エピソード毎の報酬を格納するリスト
        self.imbalance_summary = [] # エピソード毎のインバランスを格納するリスト
        self.action_difference_summary = [] # エピソード毎のアクションの差分のリスト
        self.deal_profit_summary = [] # エピソード毎の取引利益を格納するリスト
        self.episode_action_summary = []


        ## PPOで使うパラメーターの設定
        # action spaceの定義(上下限値を設定。actionは連続値。)
        # - 1time_step(ex.30 min)での充放電量(規格値[0,1])の上下限値を設定
        ##--------------------------------- 修正案 --------------------------------------##
        self.action_space = spaces.Box(low = -self.battery_max_cap * 0.5, high = self.battery_max_cap * 0.5, shape = (1, ), dtype = np.float32)
        # -2.0 ~ 2.0 [kW]の範囲
        ##-------------------------------------------------------------------------------##
        
        # 状態(observation=SoC)の上限と下限の設定
        # observation_spaceの定義(上下限値を設定。observationは連続値。)
        # - 1time_step(ex.30 min)でのSoC(規格値[0,1])の上下限値を設定
        # - PVout, price, imbalance, SoCの4つの値を使っている

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
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # 特徴量の正規化をここで行います
        # 特徴量の最大値と最小値を取得
        self.pvout_max = self.df_train['PVout'].max()
        self.pvout_min = self.df_train['PVout'].min()
        self.price_max = self.df_train['price'].max()
        self.price_min = self.df_train['price'].min()
        self.imbalance_max = self.df_train['imbalance'].max()
        self.imbalance_min = self.df_train['imbalance'].min()

        # データフレーム内で正規化　0.0 ~ 1.0
        self.df_train['PVout'] = self.normalize(self.df_train["PVout"], self.pvout_max, self.pvout_min)
        self.df_train['price'] = self.normalize(self.df_train["price"], self.price_max, self.price_min)
        self.df_train['imbalance'] = self.normalize(self.df_train["imbalance"], self.imbalance_max, self.imbalance_min)

        # # Mode選択
        # self.mode = train    # train or test
    
    # normalize function(the output typr is "Series")
    def normalize(self, series, max_val, min_val):
        if max_val - min_val == 0:
            return series
        else:
            return (series - min_val) / (max_val - min_val)
    
    # denormalize function(the output type is "float")
    def denormalize(self, normalized_series, max_val, min_val):
        if max_val - min_val == 0:
            return float(normalized_series)
        else:
            return float(normalized_series * (max_val - min_val) + min_val)
        
    # get time data
    def get_time_data(self, state_idx):
        time_of_day = state_idx % self.day_steps
        theta = 2 * np.pi * time_of_day / self.day_steps
        sin_time = np.sin(theta)
        cos_time = np.cos(theta)
        return sin_time, cos_time
    
    # check if it's the end time of the day
    def check_termination_time(self, state_idx):
        time_of_day = state_idx % self.day_steps
        if time_of_day == 0:
            terminated = True
        else:
            terminated = False
        # return terminated
        return time_of_day
        
    def operate_action(self, PV, action, current_soc):
        '''
        引数
        - PV: PV発電量の実測値[kW]
        - action: RLからのアクション, -2.0 ~ 2.0[kW](__init__()で設定)
        - current_soc: 現在のSoC, 0.0 ~ 1.0[割合]
        出力
        - edited_action: RLからのアクションを実際に動作させるために編集した値 = -2.0 ~ 2.0[kWh]
        - next_soc: 次のSoC, 0.0 ~ 1.0[割合]
        - action_difference: RLからのアクションと編集後のアクションの差分の絶対値 [kW]
        '''

        current_soc = current_soc * self.battery_max_cap # 0.0~1.0[割合]を0.0~4.0[kWh]に変換
        # 充電時
        if action < 0:
            if PV + action < 0: # PV発電よりも充電計画値が多い(PV充電エラー)
                edited_action = - PV
                _next_soc = current_soc - edited_action
                # 過剰充電(蓄電池の充放電失敗)
                if _next_soc > 4.0:
                    next_soc = 4.0
                    edited_action = current_soc - next_soc
                # 正常充電(蓄電池の充電成功)
                else:
                    next_soc = _next_soc
            else: # PV予測値内で充電(PV充電成功)
                edited_action = action
                _next_soc = current_soc - edited_action
                # 過剰充電(蓄電池の充電失敗)
                if _next_soc > 4.0:
                    next_soc = 4.0
                    edited_action = current_soc - next_soc
                # 正常充電(蓄電池の充電成功)
                else:
                    next_soc = _next_soc
        # 放電時(action >= 0)
        else: 
            # PV発電量予測値は閾値として考慮しない
            edited_action = action
            _next_soc = current_soc - edited_action
            # 過剰放電(蓄電池の放電エラー)
            if _next_soc < 0.0:
                next_soc = 0.0
                edited_action = current_soc - next_soc
            # 正常放電(蓄電池の放電成功)
            else:
                next_soc = _next_soc
        
        next_soc = next_soc / self.battery_max_cap # 0.0~4.0[kW] -> 0.0~1.0[割合]
        action_difference = abs(action - edited_action)

        return edited_action, next_soc, action_difference

    ## def reset(self)
    ## 状態の初期化: trainingで1episode(1日)終わると呼ばれる
    # - PPOのlearn(RL_train.py内にある)を走らせるとまず呼ばれる。
    # - RL_env.pyのstepメソッドにおいて、terminated = Trueになると呼ばれる。terminatedを制御することで任意のタイミングでresetを呼ぶことができる。
    # - 1episode(1日)終わるとresetを呼ぶように実装してある(def step参照)ので、次の日のデータ(48コマ全て)をstateへ入れる
    # - 現状ではdeterministicな予測を使っている -> 改良が必要。確率的になるべき。
    # PVout: PV発電量の実績値
    # price: 電力価格の実績値
    # imablance: インバランス価格の実績値
    # SoC: 前日の最終SoC soc_listの最後の要素(前episodeの最終timestep)を新しいepisodeでの初期SoCとして使う
    #### time_stepごとのactionの決定とrewardの計算を行う
    # - trainのときに使う。testのときはstep_for_testを使う
    def reset(self):
        # -------------------- リセット --------------------
        self.current_episode_reward = 0
        self.reward_list = []
        self.soc_list = []
        self.action_difference_ = 0
        self.current_imbalance = 0
        self.current_deal_profit = 0

        self.current_action_sum = 0
        # -------------------------------------------------

        ## ---------------------- ランダムな開始状態を生成 ---------------------- ##
        # データセット内の総日数を計算
        total_days = len(self.df_train) // self.day_steps
        # ランダムに日付を選択
        random_day = np.random.randint(0, total_days)
        # state_idxを選択した日付の開始インデックスに設定
        self.state_idx = random_day * self.day_steps
        # 初期SoCをランダムに設定
        initial_soc = np.random.uniform(0, 1)
        self.soc_list = [initial_soc]

        # 時間情報の計算
        sin_time, cos_time = self.get_time_data(self.state_idx)

        # 初期日付の観測値を取得
        observation = [
            # PVout, price, imbalanceは予測値であるべきでは？現在は実測値を実測値として使用している
            self.df_train["PVout"][self.state_idx], # 実測値
            self.df_train["price"][self.state_idx], # 実測値
            self.df_train["imbalance"][self.state_idx], # 実測値
            initial_soc, # 初期SoC
            sin_time, # 時間情報
            cos_time  # 時間情報
        ]
        return observation
        
    #     ##-----------------------------------要検討--------------------------------------##

    #     # -----------------------------------------------------------------------------------------------------------------
    #     # Rewardは、時系列的に後ろの方になるほど係数で小さくする必要がある。1 episode内で後ろのsteoのrewardを小さくする実装を考える
    #     # _get_rewardからの戻りrewardにgammaとstate_idxをかければ良さそう。あとで　実装する。
    #     # 必要なのか？
    #     # ------------------------------------------------------------------------------------------------------------------

    #     ##-----------------------------------要検討--------------------------------------##
    #      # SoCが0~1の範囲内に収まるようにするか、ペナルティを与えるか,、とりあえず範囲内に収まるようにする
    #     next_soc = max(0.0, min(next_soc, 1.0))
    #     ##-------------------------------------------------------------------------------##

    #     ##-----------------------------------要検討--------------------------------------##
    #     # これは何かしないといけない？
    #     # actionをnp.float32型からfloat型に変換してから足す。stable_baselines3のobservatuionの仕様のため。
    #     ##-------------------------------------------------------------------------------##

    
    def step(self, action):
        '''
        action: 充放電量, -2.0 ~ 2.0 [kW]
        '''
        # debug用
        # pvout_debug = self.df_train["PVout"][self.state_idx]
        # if self.episode_count >= 30000:
        #     if pvout_debug == 0:
        #         with open("Battery-Control-By-Reinforcement-Learning/RL_action.csv", "a") as f:
        #             f.write(f"{pvout_debug},{str(action[0])}\n")

        ## 「action > 0 -> 放電、action < 0 -> 充電」
        # time_of_day、今の時間が1日の終端状態(hour = 23.5)かどうかを判定する
        # terminated = self.check_termination_time(self.state_idx)
        # current SoCを取得
        current_soc = self.soc_list[-1] # 0.0~1.0[割合]
        edited_action, next_soc, action_difference = self.operate_action(self.df_train["PVout"][self.state_idx], action, current_soc)

        # 報酬とtruncatedを取得＆報酬をリストに追加＆現在のエピソードの合計報酬を計算
        # reward, truncated = self._get_reward(next_soc, float(action))
        reward = self._get_reward(next_soc, action_difference, edited_action)
        self.reward_list.append(reward)
        self.action_difference_ += action_difference
        self.current_episode_reward += self.reward_list[-1]

        # debug
        self.current_action_sum += edited_action

        # 終端状態(hour = 23.5)
        if self.df_train["hour"][self.state_idx] == 23.5:
            # 終端状態に達したので、現在のエピソードの合計報酬を各エピソードの報酬リストに追加
            # if self.PV_charge_error == 0:
            #     self.current_episode_reward += 1000
            # if self.charge_discharge_error == 0:
            #     self.current_episode_reward += 1000
            self.episode_rewards_summary.append(self.current_episode_reward)
            self.imbalance_summary.append(self.current_imbalance)
            self.action_difference_summary.append(self.action_difference_)
            self.deal_profit_summary.append(self.current_deal_profit)

            # debug
            self.episode_action_summary.append(self.current_action_sum)

            sin_time, cos_time = self.get_time_data(self.state_idx)
            # 終端状態の情報を登録
            observation = [
                self.df_train["PVout"][self.state_idx],
                self.df_train["price"][self.state_idx],
                self.df_train["imbalance"][self.state_idx],
                next_soc,
                sin_time,
                cos_time
            ]
            info = {"episode_reward_summary": self.current_episode_reward}
            terminated = True
            # for debug
            self.episode_count += 1

        # 終端状態でない(hour != 23.5)
        else:
            # next socをリストに追加
            self.soc_list.append(next_soc)
            # nextタイムステップに更新
            self.state_idx += 1
            # next時間情報の計算＆next観測値の取得
            sin_time, cos_time = self.get_time_data(self.state_idx)
            observation = [
                self.df_train["PVout"][self.state_idx],
                self.df_train["price"][self.state_idx],
                self.df_train["imbalance"][self.state_idx],
                next_soc,
                sin_time,
                cos_time
            ]
            # infoに情報入れると計算がくそ遅くなる
            # infoにterminatedとtruncatedを記録
            # info = {
            #     "terminated": terminated,
            #     "truncated": truncated
            # }
            info = {}
            terminated = False
        
            # デバッグ情報をリストに追加
        self.end_terminated_state.append(terminated)
        # self.end_truncated_state.append(truncated)
        
        return observation, reward, terminated, info


    ## _get_reward(self, action, next_SoC, trainOrTest)
    ## 現在の状態と行動に対するrewardを返す(1step分)
    # - rewardは1日(1 episode)ごとに合計される
    # - rewardは学習(train)の場合でしか使わない（testでは使わない）
    # - actionの単位は電力量[kWh or MWh], [-2.0~2.0]
    def _get_reward(self, next_SoC, action_difference_abs, edited_action):
        '''
        入力
        next_soc: 次のSoC, 0.0 ~ 1.0[割合]
        action_difference: RLからのアクションと編集後のアクションの差分の絶対値 [kWh]
        edited_action: RLからのアクションを実際に動作させるために編集した値 = -2.0 ~ 2.0[kWh]
        bid_action: 前日入札値[kWh]

        出力
        reward : 報酬
        '''
        ## df.trainからstate_idx(当該time_step)部分のデータを抽出し、正規化を解除
        pv_gen_normalized = self.df_train.loc[self.state_idx, "PVout"]  # PV発電実績値（正規化済み）
        pv_gen = self.denormalize(pv_gen_normalized, self.pvout_max, self.pvout_min) # PV発電実績値（非正規化）
        energyprice_normalized = self.df_train.loc[self.state_idx, "price"]  # 電力価格実績値（正規化済み）
        energyprice = self.denormalize(energyprice_normalized, self.price_max, self.price_min) # 電力価格実績値（非正規化）
        imbalance_price_normalized = self.df_train.loc[self.state_idx, "imbalance"] # インバランス価格実績値（正規化済み）
        imbalance_price = self.denormalize(imbalance_price_normalized, self.imbalance_max, self.imbalance_min) # インバランス価格実績値（非正規化）

        # Reward1: Energy Trasnfer（電力系統へ流す売電電力量）を計算する
        # deal_energyはaction + genと0の大きい方を採用する
        # PVの発電量が充電される場合はactionがマイナスになってpv_genを相殺するので、action + pv_genとする
        #  action + gen > 0 →action + gen
        #  action + gen < 0 →0
        #  action,pv_gen,deal_energyは[kWh]であることに注意

         # **予測誤差を考慮する**
        # 予測誤差をシミュレーション（平均0、標準偏差sigmaの正規分布からサンプリング）
        # forecast_error = 入札量と実際の取引量の差を表す
        sigma = 0.5  # 予測誤差の標準偏差を適切に設定
        forecast_error = np.random.normal(0, sigma)

        # PV発電量の予測値を計算
        pv_gen_forecast = pv_gen + forecast_error

        ## ------------------------------------要検討--------------------------------------##
        # 予定していたエネルギー供給量（入札量）を計算
        # このタイミングで予測値を持ってきて、入札量を計算すると、行動を丸めた意味がなくなるため、obsに実測値だけでなく予測値も入れた方がいいかも
        bid_energy = max(pv_gen_forecast + edited_action, 0)
        ## ---------------------------------------------------------------------------------##
        # 実際のエネルギー供給量を計算
        deal_energy = pv_gen + edited_action

        # インバランスエネルギーの計算（絶対値を取る）入札量と実際の取引量の差分
        imbalance_energy = abs(bid_energy - deal_energy)

        # Reward1: Energy Transfer（電力系統へ流す売電電力量）を計算する
        # 取引の純粋な利益を計算
        deal_profit = deal_energy * energyprice
        self.current_deal_profit += deal_profit

        # **インバランスコストを計算**(インバランスエネルギー×インバランス価格)
        imbalance_cost = imbalance_energy * imbalance_price

        ##-----------------------------------要検討・ペナルティ設定--------------------------------------##
        # 制約条件のペナルティを追加
        # penalty = 0
        # penalty_weight = 100  # ペナルティの重み
        # # Reward2: バッテリー充電がPV発電量を超える場合のペナルティ
        # if action < 0 and abs(action) > pv_gen:
        #     penalty -= penalty_weight * (abs(action) - pv_gen)  # 超過量に比例したペナルティ
        # # Reward3: バッテリー放電がnext_SoCを超える場合のペナルティ
        # if action > 0 and action > next_SoC:
        #     penalty -= penalty_weight * (action - next_SoC)  # 超過量に比例したペナルティ  
        # # Reward4: next_SoCが0〜1の範囲外になる場合のペナルティ
        # if next_SoC < 0:
        #     penalty -= penalty_weight * abs(next_SoC)  # SoCが0未満の場合のペナルティ
        # elif next_SoC > 1:
        #     penalty -= penalty_weight * (next_SoC - 4)  # SoCが1を超えた場合のペナルティ
        ##---------------------------------------------------------------------------------------------##

        # ADをペナルティに設定
        penalty = action_difference_abs * imbalance_price # imbalancepriceの方が良い？
        # **インバランスコストを考慮した最終報酬**
        self.current_imbalance -= imbalance_cost
        reward = - penalty
        return reward