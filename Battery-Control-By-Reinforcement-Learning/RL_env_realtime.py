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

        # 予測値の精度(bid and realtime)
        self.sigma_realtime = 0.1 # ex) 0.1 = 10%誤差 
        self.upper_times = 2.0 # 予測値の上限値の倍率
        self.lower_times = 0.0 # 予測値の下限値の倍率


        ## reset()で初期化---------------------------------------------------------
        self.bid_deal_list = [] # 仮想のbid計画での充放電量
        self.soc_list_bid = [] # 仮想のbid計画でのSoC
        self.soc_list_realtime_actual = [] # 実際のSoC
        self.bid_deal_list = [] # 仮想のbid計画での充放電量
        self.current_episode_reward = 0 # 現在のエピソードの合計報酬
        self.reward_list = [] # 現在のエピソードでの各stepのreward
        self.state_idx = 0 # time step
        self.current_imbalance = 0 # エピソードの合計インバランス
        self.action_difference = 0 # RLからのアクションと実際のアクションの差分
        self.current_deal_profit = 0 # エピソードの合計取引利益
        self.bid_deal_normalized = 0 # 仮想のbid計画での入札値 [正規化]
        self.pv_realtime_predict = 0 # PV発電量の予測値
        self.price_realtime_predict = 0 # 電力価格の予測値
        self.imbalance_realtime_predict = 0 # インバランス価格の予測値
        ## ------------------------------------------------------------------------
        self.episode_rewards_summary = []  # エピソード毎の報酬を格納するリスト
        self.imbalance_summary = [] # エピソード毎のインバランスを格納するリストト
        self.action_difference_summary = [] # エピソード毎のアクションの差分のリスト
        self.deal_profit_summary = [] # エピソード毎の取引利益を格納するリスト

        ## PPOで使うパラメーターの設定
        # action spaceの定義(上下限値を設定。actionは連続値。)
        # - 1time_step(ex.30 min)での充放電量(規格値[0,1])の上下限値を設定
        # -2.0 ~ 2.0 [kW]の範囲
        self.action_space = spaces.Box(low = -self.battery_max_cap * 0.5, high = self.battery_max_cap * 0.5, shape = (1, ), dtype = np.float32)

        # 状態(observation=SoC)の上限と下限の設定
        # observation_spaceの定義(上下限値を設定。observationは連続値。)
        # - 1time_step(ex.30 min)でのSoC(規格値[0,1])の上下限値を設定
        # - PVout, price, imbalance, SoCの4つの値を使っている

        # 観測空間の次元数を計算（既存の4つ + sin_time + cos_time）
        obs_dim = 5 + 2  # リアルタイム予測値(PVout, Price, Imbalance), bid入札値, SoC, sin_time, cos_time

        # 観測空間の下限と上限を設定
        low = np.array([-np.inf] * obs_dim)
        high = np.array([np.inf] * obs_dim)

        # リアルタイム予測値＆bid入札値&SoCの範囲を設定（0から1）
        low[0:5] = 0
        high[0:5] = 1.0
        # sin_timeとcos_timeの範囲を設定（-1から1）
        low[5:7] = -1.0
        high[5:7] = 1.0

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
        self.df_train['PVout_normalized'] = self.normalize(self.df_train["PVout"], self.pvout_max, self.pvout_min)
        self.df_train['price_normalized'] = self.normalize(self.df_train["price"], self.price_max, self.price_min)
        self.df_train['imbalance_normalized'] = self.normalize(self.df_train["imbalance"], self.imbalance_max, self.imbalance_min)

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
    
    # 誤差を考慮したリアルタイム予測値を取得する関数 
    def truncated_normal_predict(self, sigma, name, max, min):
        '''
        引数
        - sigma: 予測値の標準偏差、0.0 ~ 1.0, init()で設定

        出力
        - dist.rvs(): トランケート正規分布に従うランダムな値(lower ~ upperの範囲内から取得)
        - 実測値に誤差が加えられた、予測値が出力される[kWh]
        '''
        lower = self.lower_times * min #[kWh] 実質0
        upper = self.upper_times * max #[kWh]

        # 実測値を平均にする
        mean = self.df_train.at[self.state_idx, name]

        dist = truncnorm(
            (lower - mean) / sigma, # 下限の標準化
            (upper - mean) / sigma, # 上限の標準化
            loc = mean, # 平均
            scale = sigma * mean # 標準偏差
        )
        predict_value = float(dist.rvs())
        predict_value_normalized = self.normalize(predict_value, upper, lower)
        return predict_value, predict_value_normalized
    
    # 行動がPV & 蓄電池から見て適正範囲に編集する関数
    def operate_action(self, PV, action, current_soc):
        '''
        引数
        - PV: PV発電量の実測値 or 予測値[kW]
        - action: RLからのアクション, -2.0 ~ 2.0[kW](__init__()で設定) → 予測値をもとにした行動 or 予測値を基にした行動を編集した行動
        - current_soc: 現在のSoC, 0.0 ~ 1.0[正規化]
        出力
        - edited_action: RLからのアクションを実際に動作させるために編集した値 = -2.0 ~ 2.0[kWh]
        - next_soc: 次のSoC, 0.0 ~ 1.0[正規化]
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

        return edited_action, next_soc
    
    # 仮想のbid入札量＆次のSoCを取得する関数
    def get_bid_deal(self):
        # PV発電量実績値の取得
        pv_actual = self.df_train.at[self.state_idx, "PVout"]  # [kWh]
        # PV実測値にノイズを足して(bid)PV予測値を作成
        pv_bid_predict = pv_actual + np.random.uniform(-self.pvout_min, (self.upper_times * self.pvout_max)) # 0 ~ pvout_max(1+upper_times)[kWh]
        # PV予測値の正規化
        pv_bid_predict_normalized = self.normalize(pv_bid_predict, self.pvout_max * (1 + self.upper_times), 0) # 0.0~1.0[正規化]
        # bid充放電計画量
        bid_action = np.random.uniform(-self.battery_max_cap * 0.5, self.battery_max_cap * 0.5)  # -2.0 ~ 2.0[kWh]
        # 充放電が適切か判定し編集
        edited_bid_action, next_soc_bid = self.operate_action(pv_bid_predict, bid_action, self.soc_list_bid[-1])
        # bid入札量
        bid_deal = pv_bid_predict + edited_bid_action # [kWh]
        bid_deal_normalized = self.normalize(bid_deal, self.pvout_max*(1 + self.upper_times) + (self.battery_max_cap * 0.5), 0) # [正規化]
        # bid入札量を記録
        self.bid_deal_list.append(bid_deal)
        # bid計画での次の時間のSoCを記録 -> 疑似的に1日を通したbid計画を再現できそう
        self.soc_list_bid.append(next_soc_bid) # RLモデルが観測しないためobsには入れない

        return bid_deal_normalized

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
        self.soc_list_bid = []
        self.soc_list_realtime_actual = []
        self.action_difference = 0
        self.current_imbalance = 0
        self.current_deal_profit = 0
        self.pv_realtime_predict = 0
        self.price_realtime_predict = 0
        self.imbalance_realtime_predict = 0
        self.bid_deal_normalized = 0
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
        self.soc_list_bid = [initial_soc]
        self.soc_list_realtime_actual = [initial_soc]

        ## ------------------------------ ↓要検討↓ ------------------------------ ##
        # bid入札値の取得[kWh] & [正規化]
        self.bid_deal_normalized = self.get_bid_deal()

        ## リアルタイム予測値の取得[kWh] & [正規化]
        self.pv_realtime_predict, self.pv_realtime_predict_normalized = self.truncated_normal_predict(self.sigma_realtime, "PVout", self.pvout_max, self.pvout_min)
        self.price_realtime_predict, self.price_realtime_predict_normalized = self.truncated_normal_predict(self.sigma_realtime, "price", self.price_max, self.price_min)
        self.imbalance_realtime_predict, self.imbalance_realtime_predict_normalized = self.truncated_normal_predict(self.sigma_realtime, "imbalance", self.imbalance_max, self.imbalance_min)
        ## ------------------------------ ↑要検討↑ ------------------------------ ##

        # 時間情報の計算
        sin_time, cos_time = self.get_time_data(self.state_idx)

        # 初期日付の観測値を取得
        # リアルタイム予測値(PVout, price, imbalance)、前日入札値、初期SoC、時間情報(sin, cos) -> 7個の観測値
        observation = [
            # 全て正規化されている
            self.pv_realtime_predict_normalized, # PV発電量の予測値[正規化]
            self.price_realtime_predict_normalized, # 電力価格の予測値[正規化]
            self.imbalance_realtime_predict_normalized, # インバランス価格の予測値[正規化]
            self.bid_deal_normalized, # bid入札値[正規化]
            initial_soc, # 初期SoC[正規化]
            sin_time, # 時間情報[正規化]
            cos_time # 時間情報[正規化]
        ]
        return observation
     
    #     ##-----------------------------------要検討--------------------------------------##

    #     # -----------------------------------------------------------------------------------------------------------------
    #     # Rewardは、時系列的に後ろの方になるほど係数で小さくする必要がある。1 episode内で後ろのsteoのrewardを小さくする実装を考える
    #     # _get_rewardからの戻りrewardにgammaとstate_idxをかければ良さそう。あとで　実装する。
    #     # ------------------------------------------------------------------------------------------------------------------

    #     ## ここでsoc_list[-1]を使ってrewardを計算すると、(a0, s1)の報酬が出てきてしまう。これは正しいのか？本当は(a0, s0)の報酬が欲しいのでは？

    #     # soc_listの最後の要素(前timestepのSoC)にactionを足す
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
        current_soc = self.soc_list_realtime_actual[-1] # 0.0~1.0[正規化]

        # PV予測値 & 現在の蓄電池容量に対して、actionを編集
        '''
        edited_action: RL行動->編集後の行動 [kWh]
        next_soc: 編集後の行動をした場合の、次のSoC [正規化]
        '''
        edited_action, next_soc = self.operate_action(self.pv_realtime_predict, action, current_soc)

        # 予測値に対して編集した行動が実測値に対して適切かどうか、actionを編集
        '''
        edited_action_actual: 編集後の行動->真の行動[kWh]
        next_soc_actual: 真の行動をした場合の、次状態[正規化]
        '''
        edited_action_actual, next_soc_actual = self.operate_action(self.df_train.at[self.state_idx, "PVout"], edited_action, current_soc)

        # 報酬を取得＆報酬をリストに追加＆現在のエピソードの合計報酬を計算
        reward = self._get_reward(next_soc, action, edited_action_actual)

        # 終端状態(hour = 23.5)
        if self.df_train.at[self.state_idx, "hour"] == 23.5:
            # 終端状態に達したので、現在のエピソードの合計報酬を各エピソードの報酬リストに追加
            self.episode_rewards_summary.append(self.current_episode_reward)
            self.imbalance_summary.append(self.current_imbalance)
            self.action_difference_summary.append(self.action_difference)
            self.deal_profit_summary.append(self.current_deal_profit)
            # 時間情報の計算
            sin_time, cos_time = self.get_time_data(self.state_idx)
            # 終端状態の情報を登録
            observation = [
                self.pv_realtime_predict_normalized, # PV発電量の予測値[正規化]
                self.price_realtime_predict_normalized, # 電力価格の予測値[正規化]
                self.imbalance_realtime_predict_normalized, # インバランス価格の予測値[正規化]
                self.bid_deal_normalized, # bid入札値[正規化]
                next_soc, # 次のSoC[正規化]
                sin_time, # 時間情報[正規化]
                cos_time # 時間情報[正規化]
            ]
            info = {"episode_reward_summary": self.current_episode_reward}
            terminated = True
            # for debug
            self.episode_count += 1

        # 終端状態でない(hour != 23.5)
        else:
            self.soc_list_realtime_actual.append(next_soc_actual)
            # 次の時間に進む
            self.state_idx += 1
            # 次の時間のbid入札値を取得
            self.bid_deal_normalized = self.get_bid_deal()
            # 次の時間のrealtime予測値を取得
            self.pv_realtime_predict, self.pv_realtime_predict_normalized = self.truncated_normal_predict(self.sigma_realtime, "PVout", self.pvout_max, self.pvout_min)
            self.price_realtime_predict, self.price_realtime_predict_normalized = self.truncated_normal_predict(self.sigma_realtime, "price", self.price_max, self.price_min)
            self.imbalance_realtime_predict, self.imbalance_realtime_predict_normalized = self.truncated_normal_predict(self.sigma_realtime, "imbalance", self.imbalance_max, self.imbalance_min)
            # next時間情報の計算＆next観測値の取得
            sin_time, cos_time = self.get_time_data(self.state_idx)
            
            # 次の時間の情報
            observation = [
                self.pv_realtime_predict_normalized, # PV発電量の予測値[正規化]
                self.price_realtime_predict_normalized, # 電力価格の予測値[正規化]
                self.imbalance_realtime_predict_normalized, # インバランス価格の予測値[正規化]
                self.bid_deal_normalized, # bid入札値[正規化]
                next_soc, # 次のSoC[正規化]
                sin_time, # 時間情報[正規化]
                cos_time # 時間情報[正規化]
            ]

            # infoに情報入れると計算がくそ遅くなる
            info = {}
            terminated = False
        
        return observation, reward, terminated, info


    ## _get_reward(self, action, next_SoC, trainOrTest)
    ## 現在の状態と行動に対するrewardを返す(1step分)
    # - rewardは1日(1 episode)ごとに合計される
    # - rewardは学習(train)の場合でしか使わない（testでは使わない）
    # - actionの単位は電力量[kWh or MWh], [-2.0~2.0]
    def _get_reward(self, next_SoC, action, edited_action_actual):
        '''
        入力
        next_soc: 次のSoC, 0.0 ~ 1.0[正規化], 現状使わない
        action: RLモデルの行動, -2.0 ~ 2.0[kWh]
        edited_action_actual: RLからのアクションを実際に動作させるために編集した値, -2.0 ~ 2.0[kWh]
        bid_deal: bid入札値[kWh]

        出力
        reward : 報酬
        '''
        # 実績値を取得
        pv_actual = self.df_train.at[self.state_idx, "PVout"]  # PV発電実績値[kWh]
        energyprice_actual = self.df_train.at[self.state_idx, "price"] # 電力価格実績値[Yen/kWh]
        imbalance_price_actual = self.df_train.at[self.state_idx, "imbalance"] # インバランス価格実績値[Yen/kWh]
        # bid入札値を取得
        bid_deal = self.bid_deal_list[-1] # [kWh]

        # 実際の取引量
        deal_actual = pv_actual + edited_action_actual # [kWh]
        # 取引誤差
        deal_error = abs(bid_deal - deal_actual) # [kWh]
        # 取引利益
        deal_profit =  deal_actual * energyprice_actual # [Yen]
        self.current_deal_profit += deal_profit
        # 取引損失
        deal_loss = deal_error * imbalance_price_actual # [Yen]
        self.current_imbalance -= deal_loss
        # RL行動と真の行動の差分
        _action_difference = abs(action - edited_action_actual) # [kWh]
        self.action_difference += _action_difference
        # ペナルティ
        penalty = _action_difference * imbalance_price_actual # [Yen], energyprice_actualでもいいかも
        # 報酬
        reward = - deal_loss - penalty
        # 報酬をリストに追加
        self.reward_list.append(reward)
        # 現在のエピソードの合計報酬を計算
        self.current_episode_reward += reward

        return reward
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