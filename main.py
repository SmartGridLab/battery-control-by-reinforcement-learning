# Import
import gym
import warnings
import numpy as np
import pandas as pd
import math as ma
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import A2C, PPO2
from gym import spaces
from scipy.stats import norm

warnings.simplefilter('ignore')

# %%
#パラメータ(学習条件などは以下のパラメータを変更するだけで良い)
num_episodes = 48 # 1日のコマ数(固定)
pdf_day = 30 #確率密度関数作成用の日数
train_days = 30 # 学習日数
test_day = 58 # テスト日数＋１
episode = 10000  # 学習回数
PV_parameter = "PVout_true" # Forecast or PVout_true (学習に使用するPV出力値の種類)
mode = "learn" # learn or test
model_name = "ESS_learn" # ESS_learn ESS_learn_1000

# 環境設定
env = ESS_Model(mode, pdf_day, train_days, test_day, PV_parameter)
env.main_root(mode, num_episodes, train_days, episode, model_name)

# %%
# 利益、インバランス料金の計算
import numpy as np
import pandas as pd
import math as ma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#実際の制御に基づく利益
def graph(y1, x_label, y_label, label_name):
        episode = len(PVout_true_PV_real)/48 # テスト日数
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.rcParams["font.size"] = 35
        plt.plot(np.arange(episode), y1, label = label_name)
        plt.legend(prop={"family":"MS Gothic"})
        plt.xlabel(x_label, fontname="MS Gothic")
        plt.ylabel(y_label, fontname="MS Gothic")
        plt.close()
        
        return fig

def graph_all(y1, y2, y3, x_label, y_label, label_name_1, label_name_2, label_name_3):
        episode = len(PVout_true_PV_real)/48 # テスト日数
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.rcParams["font.size"] = 30
        plt.plot(np.arange(episode), y1, label = label_name_1)
        plt.plot(np.arange(episode), y2, label = label_name_2)
        plt.plot(np.arange(episode), y3, label = label_name_3)
        plt.legend(prop={"family":"MS Gothic"})
        plt.xlabel(x_label, fontname="MS Gothic")
        plt.ylabel(y_label, fontname="MS Gothic")
        plt.close()
        
        return fig

train_day = 60
test_day = 58

DATA = pd.read_csv("PVout_true_generation.csv")
PVout_true_PV_real = DATA["PVout_true_PV_real"]
PVout_true_ESS_real = DATA["PVout_true_charge_discharge_real"]
DATA = pd.read_csv("Forecast_generation.csv")
Forecast_PV_real = DATA["Forecast_PV_real"]
Forecast_ESS_real = DATA["Forecast_charge_discharge_real"]

DATA = pd.read_csv("train_and_test_data.csv", encoding="shift-jis")
PV_out = DATA["PVout_true"]
PV_out_true = PV_out[48*train_day:48*(train_day + test_day)]
price = DATA["price[yen/kW30m]"]
price_data = price[48*train_day:48*(train_day + test_day)]

PV_out_true = (PV_out_true.values)# 型変換
PV_out_true = PV_out_true.reshape((len(PV_out_true), 1)) 
price_data = (price_data.values)# 型変換
price_data = price_data.reshape((len(price_data), 1)) 

battery = 0
ESS_profit = 0
PV_profit = 0
total_profit = 0
time = 0
all_ESS_profit = []
all_PV_profit = []
all_total_profit = []
all_ESS = []
all_battery = []
all_price = []
all_PV = []
all_forecast_out = []

for i in range(0, len(PVout_true_PV_real)):
    time += 1
    PVout_true_ESS_time = PVout_true_ESS_real[i]
    Forecast_ESS_time_real = Forecast_ESS_real[i]
    PV_out_time = PV_out_true[i]
    price_time = price_data[i]
    Forecast_PV_real_time = Forecast_PV_real[i]

    if PV_out_time < 0:
        PV_out_time = np.array([0.0])
    #forecast
        #充電
    if Forecast_ESS_time_real < 0: 
        Forecast_total_real_time = Forecast_PV_real_time # 合計出力の予定値
        if PV_out_time + Forecast_ESS_time_real < 0: # pv発電量を全て充電に使用
            Forecast_ESS_time_real = -1*PV_out_time
            PV_sell = [0]
            battery = battery - Forecast_ESS_time_real/2
            PV_profit += PV_sell*price_time
            forecast_out_true = PV_sell[0]
            all_ESS.append(Forecast_ESS_time_real[0])
            all_PV.append(PV_sell[0])
            all_forecast_out.append(forecast_out_true)
        elif PV_out_time + Forecast_ESS_time_real >= 0 : # pv発電量に余裕があるとき
            battery = battery - Forecast_ESS_time_real/2
            if battery > 4:
                battery = [4]
            PV_sell = PV_out_time + Forecast_ESS_time_real
            forecast_out_true = PV_sell[0]
            if forecast_out_true > Forecast_total_real_time: # 真値 > 予定値 (余剰)
                plus = forecast_out_true - Forecast_total_real_time # 余剰分
                if battery[0] >= 4:
                    plus_gen = plus # 余剰分を売電
                elif battery[0] < 4:
                    battery = battery + plus/2 # 余剰分を充電
                    if battery[0] >= 4: # 満充電になる時
                        out_battery = battery
                        battery = [4]
                        plus_gen = out_battery - battery[0] # 充電しても使い切れなかった分
                        out_gen = plus - plus_gen
                        PV_sell = PV_sell - out_gen
                    elif battery[0] < 4:
                        plus_gen = 0 # 充電分で使い切った時 
                        out_gen = plus - plus_gen
                        PV_sell = PV_sell - out_gen
                out_gen = plus - plus_gen
                forecast_out_true = PV_sell[0]
            ESS_profit += 0
            PV_profit += PV_sell*price_time
            all_ESS.append(Forecast_ESS_time_real)
            all_PV.append(PV_sell[0])
            all_forecast_out.append(forecast_out_true)

        #放電
    elif Forecast_ESS_time_real >= 0:
        Forecast_total_real_time = Forecast_ESS_time_real + Forecast_PV_real_time # 合計出力の予定値
        if battery - Forecast_ESS_time_real/2 < 0:
            Forecast_ESS_time_real = battery
            battery = [0]
            PV_sell = PV_out_time
            forecast_out_true = Forecast_ESS_time_real[0] + PV_sell[0]
            if forecast_out_true >= Forecast_total_real_time: # 真値 > 予定値 (余剰)
                plus = forecast_out_true - Forecast_total_real_time # 余剰分
                battery = battery + plus/2 # 余剰分は放電しない
                Forecast_ESS_time_real = Forecast_ESS_time_real - plus # 放電量を抑える
                forecast_out_true = forecast_out_true - plus
                all_forecast_out.append(forecast_out_true)
            elif forecast_out_true < Forecast_total_real_time: # 真値 < 予定値 (不足)
                plus = Forecast_total_real_time - forecast_out_true # 不足分
                if battery - plus/2 < 0:
                    out_gen = battery
                    battery = [0]
                elif battery - plus/2 >= 0:
                    out_gen = plus
                    battery = battery - plus/2
                Forecast_ESS_time_real = [Forecast_ESS_time_real[0] + out_gen[0]] # 放電量を増加
                forecast_out_true = forecast_out_true + out_gen
                all_forecast_out.append(forecast_out_true[0])
            ESS_profit += Forecast_ESS_time_real*price_time
            PV_profit += PV_sell*price_time
            all_ESS.append(Forecast_ESS_time_real[0])
            all_PV.append(PV_sell[0])
        elif battery - Forecast_ESS_time_real/2 >= 0:
            battery = battery - Forecast_ESS_time_real/2
            PV_sell = PV_out_time
            forecast_out_true = PV_sell[0] + Forecast_ESS_time_real
            if forecast_out_true >= Forecast_total_real_time: # 真値 > 予定値 (余剰)
                plus = forecast_out_true - Forecast_total_real_time # 余剰分
                battery = battery + plus/2 # 余剰分を充電
                if battery >= 4: # 満充電になる時
                    out_battery = battery
                    battery = [4]
                    plus_gen = out_battery[0] - battery[0] # 充電しても使い切れなかった分
                elif battery < 4:
                    plus_gen = 0 # 充電分で使い切った時
                out_gen = plus - plus_gen
                Forecast_ESS_time_real = Forecast_ESS_time_real - out_gen
                forecast_out_true = forecast_out_true - out_gen
            elif forecast_out_true < Forecast_total_real_time: # 真値 < 予定値 (不足)
                plus = Forecast_total_real_time - forecast_out_true # 不足分
                if battery - plus/2 < 0:
                    out_gen = battery[0]
                    battery = [0]
                elif battery - plus/2 >= 0:
                    out_gen = plus
                    battery = battery - plus/2
                Forecast_ESS_time_real = Forecast_ESS_time_real + out_gen # 放電量を増加
                forecast_out_true = forecast_out_true + out_gen
            ESS_profit += Forecast_ESS_time_real*price_time
            PV_profit += PV_sell*price_time
            all_ESS.append(Forecast_ESS_time_real)
            all_PV.append(PV_sell[0])
            all_forecast_out.append(forecast_out_true)

    if time == 48:
        time = 0
        total_profit = ESS_profit + PV_profit
        all_ESS_profit.append(ESS_profit[0])
        all_PV_profit.append(PV_profit[0])
        all_total_profit.append(total_profit[0])

all_ESS = pd.DataFrame(np.ravel(all_ESS))
all_PV = pd.DataFrame(np.ravel(all_PV))
all_forecast_out = pd.DataFrame(np.ravel(all_forecast_out))
data = pd.concat([all_ESS,all_PV,all_forecast_out], axis=1)
label_name = ["ESS_forecast","PV_forecast","forecast_total"]
data.columns = label_name
data.to_csv("true_data.csv")

pdf_name = "result-profit-imb.pdf"
pp = PdfPages(pdf_name) # PDFの作成
profit_graph = graph(all_total_profit,"日", "売上[円]", "売上")
pp.savefig(profit_graph)

DATA = pd.read_csv("true_data.csv")
true_PV_forecast = DATA["PV_forecast"]
true_ESS_forecast = DATA["ESS_forecast"]
true_total_forecast = DATA["forecast_total"]

DATA = pd.read_csv("forecast_generation.csv")
Forecast_PV = DATA["Forecast_PV"]
Forecast_ESS = DATA["Forecast_charge_discharge"]
Forecast_PV_real = DATA["Forecast_PV_real"]
Forecast_ESS_real = DATA["Forecast_charge_discharge_real"]
alpha_data = DATA["Forecast_alpha"]
beta_data = DATA["Forecast_beta"]

imbalance_forecast = 0
imb_Forecast_total = []
K = 1.46
L = 0.43
time = 0

for i in range(0, len(DATA)):
        time += 1
        true_PV_forecast_time = true_PV_forecast[i]
        true_ESS_forecast_time = true_ESS_forecast[i]
        Forecast_ESS_time = Forecast_ESS[i]
        Forecast_PV_time = Forecast_PV[i]
        Forecast_ESS_time_real = Forecast_ESS_real[i]
        Forecast_PV_time_real = Forecast_PV_real[i]
        alpha = alpha_data[i]
        beta = beta_data[i]

        true_total_forecast_time = true_total_forecast[i]
        if type(true_total_forecast_time) == str:
                true_total_forecast_time = float(true_total_forecast_time)

        if Forecast_ESS_time_real > 0:
                total_forecast_time_real = Forecast_PV_time_real + Forecast_ESS_time_real
        elif Forecast_ESS_time_real <= 0:
                total_forecast_time_real = Forecast_PV_time_real

        # Forecast
        # 不足
        if true_total_forecast_time < total_forecast_time_real:                
                imbalance_forecast -= (alpha + beta + K)*(abs(true_total_forecast_time - total_forecast_time_real))
        # 余剰
        elif true_total_forecast_time >= total_forecast_time_real:
                imbalance_forecast -= (alpha + beta - L)*(abs(true_total_forecast_time - total_forecast_time_real))

        if time == 48:
                imb_Forecast_total.append(imbalance_forecast)
                time = 0

imb_Graph = graph(imb_Forecast_total, "日", "インバランス料金[円]", "インバランス料金")
all_total_profit = np.array(all_total_profit)
imb_Forecast_total = np.array(imb_Forecast_total)
plus = all_total_profit+imb_Forecast_total
total_Graph = graph(plus, "日", "利益[円]", "利益")
all_Graph = graph_all(all_total_profit,imb_Forecast_total,plus,"日", "金額[円]", "売上","インバランス料金","利益")

pp.savefig(imb_Graph)
pp.savefig(total_Graph)
pp.savefig(all_Graph)
pp.close()

print(total_profit)
print((-1)*imbalance_forecast)
print(total_profit + imbalance_forecast)