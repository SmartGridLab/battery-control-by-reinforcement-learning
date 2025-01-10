import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import os

# CSVファイルの読み込み
df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

# 正確なDatetimeインデックスの生成
df['Datetime'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce') + pd.to_timedelta(df['hour'], unit='h')
df.set_index('Datetime', inplace=True)

# グラフを保存する新しいディレクトリを作成
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = f"Battery-Control-By-Reinforcement-Learning/for_debug/RLTest/RLTest_info_{current_time}"
os.makedirs(folder_path, exist_ok=True)

# グラフのスタイル設定
color_actual_realtime = "blue"
color_realtime = "green"
color_actual_bid = "red"
color_bid = "coral"
color_actual = "black"
linestyle_actual_realtime = '-.'
linestyle_realtime = ':'
linestyle_actual_bid = '--'
linestyle_bid = '-'
linestyle_actual = '--'

# プロットの作成
plt.figure(figsize=(30, 25))  # グラフのサイズを適切に調整

# energyprofitのグラフ -----------------------
plt.subplot(3,1,1)
plt.plot(df.index, df["energyprofit_actual_realtime[Yen]"], label="Energy Profit Actual Realtime [Yen]", color = color_actual_realtime, linestyle = linestyle_actual_realtime)
plt.plot(df.index, df["energyprofit_realtime[Yen]"], label="Energy Profit Realtime [Yen]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df["energyprofit_actual_bid[Yen]"], label="Energy Profit Actual Bid [Yen]", color = color_actual_bid, linestyle = linestyle_actual_bid)
plt.plot(df.index, df["energyprofit_bid[Yen]"], label="Energy Profit Bid [Yen]", color = color_bid, linestyle = linestyle_bid)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [Yen]")
plt.title("Energy Profit")
plt.grid(True)
plt.legend(loc = "upper right")

# imbalancepenaltyのグラフ -----------------------
plt.subplot(3, 1, 2)
plt.plot(df.index, df["imbalancepenalty_actual_realtime[Yen]"], label="Imbalancepenalty Actual Realtime [Yen]", color = color_actual_realtime, linestyle = linestyle_actual_realtime)
plt.plot(df.index, df["imbalancepenalty_realtime[Yen]"], label="Imbalancepenalty Realtime [Yen]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df["imbalancepenalty_actual_bid[Yen]"], label="Imbalancepenalty Actual Bid [Yen]", color = color_actual_bid, linestyle = linestyle_actual_bid)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [Yen]")
plt.title("Imbalance Penalty")
plt.grid(True)
plt.legend(loc = "upper right")

# totalprofit_actual_realtimeの累積和のグラフ -----------------------
df['totalprofit_actual_realtime_cumsum[Yen]'] = df['totalprofit_actual_realtime[Yen]'].cumsum()
df['totalprofit_realtime_cumsum[Yen]'] = df['totalprofit_realtime[Yen]'].cumsum()
df['totalprofit_actual_bid_cumsum[Yen]'] = df['totalprofit_actual_bid[Yen]'].cumsum()
df['totalprofit_bid_cumsum[Yen]'] = df['totalprofit_bid[Yen]'].cumsum()

plt.subplot(3, 1, 3)
plt.plot(df.index, df['totalprofit_actual_realtime_cumsum[Yen]'], label="Total Profit Actual Realtime Cumulative Sum [Yen]", color = color_actual_realtime, linestyle = linestyle_actual_realtime)
plt.plot(df.index, df['totalprofit_realtime_cumsum[Yen]'], label="Total Profit Realtime Cumulative Sum [Yen]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df['totalprofit_actual_bid_cumsum[Yen]'], label="Total Profit Actual Bid Cumulative Sum [Yen]", color = color_actual_bid, linestyle = linestyle_actual_bid)
plt.plot(df.index, df['totalprofit_bid_cumsum[Yen]'], label = "Total Profit Bid Cumulative Sum [Yen]", color = color_bid, linestyle = linestyle_bid)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Cumulative Value [Yen]")
plt.title("Total Profit Cumulative Sum")
plt.grid(True)
plt.legend(loc="upper right")

# グラフの表示と保存
# plt.xticks(rotation=45)  # ラベルの回転
plt.tight_layout()  # レイアウトの自動調整

plt.savefig(f"{folder_path}/Energyprofit_{current_time}.png")
plt.show()

# ------------------------------ PV発電量（予測値 and 実測値）-----------------------------------------
# プロットの作成
plt.figure(figsize=(30, 25))  # グラフのサイズを適切に調整

# PV -----------------------
plt.subplot(3, 1, 1)
plt.plot(df.index, df["PV_predict_bid[kW]"], label="PV Predict Bid [kW]", color = color_bid, linestyle = linestyle_bid)
plt.plot(df.index, df["PV_predict_realtime[kW]"], label="PV Predict Realtime [kW]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df["PV_actual[kW]"], label="PV Actual [kW]", color = color_actual, linestyle = linestyle_actual)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [kW]")
plt.title("PV Generation")
plt.grid(True)
plt.legend(loc = "upper right")

# Imbalance
plt.subplot(3, 1, 2)
plt.plot(df.index, df["imbalanceprice_predict_bid[Yen/kWh]"], label="Imbalanceprice Predict Bid [Yen/kWh]", color = color_bid, linestyle = linestyle_bid)
plt.plot(df.index, df["imbalanceprice_predict_realtime[Yen/kWh]"], label="Imbalanceprice Predict Realtime [Yen/kWh]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df["imbalanceprice_actual[Yen/kWh]"], label="Imbalanceprice Actual [Yen/kWh]", color = color_actual, linestyle = linestyle_actual)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [Yen/kWh]")
plt.title("Imbalance Price")
plt.grid(True)
plt.legend(loc = "upper right")

# Energy price
plt.subplot(3, 1, 3)
plt.plot(df.index, df["energyprice_predict_bid[Yen/kWh]"], label="Energyprice Predict Bid [Yen/kWh]", color = color_bid, linestyle = linestyle_bid)
plt.plot(df.index, df["energyprice_predict_realtime[Yen/kWh]"], label="Energyprice Predict Realtime [Yen/kWh]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df["energyprice_actual[Yen/kWh]"], label="Energyprice Actual [Yen/kWh]", color = color_actual, linestyle = linestyle_actual)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [Yen/kWh]")
plt.title("Energy Price")
plt.grid(True)
plt.legend(loc = "upper right")

# グラフの表示と保存
# plt.xticks(rotation=45)  # ラベルの回転
plt.tight_layout()  # レイアウトの自動調整

plt.savefig(f"{folder_path}/testobs_{current_time}.png")
plt.show()

# ------------------------------ 蓄電池の状態のグラフ ----------------------------------------
# プロットの作成
plt.figure(figsize=(30, 25))  # グラフのサイズを適切に調整
# SoC[%]
plt.subplot(3, 1, 1)
plt.plot(df.index, df["SoC_bid[%]"], label="SoC Bid [%]", color = color_bid, linestyle = linestyle_bid)
plt.plot(df.index, df["SoC_actual_bid[%]"], label="SoC Actual Bid [%]", color = color_actual_bid, linestyle = linestyle_actual_bid)
plt.plot(df.index, df["SoC_realtime[%]"], label="SoC Realtime [%]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df["SoC_actual_realtime[%]"], label="SoC Actual Realtime [%]", color = color_actual_realtime, linestyle = linestyle_actual_realtime)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [%]")
plt.title("State of Charge")
plt.grid(True)
plt.legend(loc = "upper right")

# Charge/Discharge Power[kW]
plt.subplot(3, 1, 2)
plt.plot(df.index, df["charge/discharge_bid[kWh]"], label="Charge/Discharge Bid [kWh]", color = color_bid, linestyle = linestyle_bid)
plt.plot(df.index, df["charge/discharge_actual_bid[kWh]"], label="Charge/Discharge Actual Bid [kWh]", color = color_actual_bid, linestyle = linestyle_actual_bid)
plt.plot(df.index, df["charge/discharge_realtime[kWh]"], label="Charge/Discharge Realtime [kWh]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df["charge/discharge_actual_realtime[kWh]"], label="Charge/Discharge Actual Realtime [kWh]", color = color_actual_realtime, linestyle = linestyle_actual_realtime)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Power [kW]")
plt.title("Charge/Discharge Power")
plt.grid(True)
plt.legend(loc = "upper right")

# EnergyTransfer Power[kW]
plt.subplot(3, 1, 3)
plt.plot(df.index, df["energytransfer_bid[kWh]"], label="EnergyTransfer Bid [kWh]", color = color_bid, linestyle = linestyle_bid)
plt.plot(df.index, df["energytransfer_actual_bid[kWh]"], label="EnergyTransfer Actual Bid [kWh]", color = color_actual_bid, linestyle = linestyle_actual_bid)
plt.plot(df.index, df["energytransfer_realtime[kWh]"], label="EnergyTransfer Realtime [kWh]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df["energytransfer_actual_realtime[kWh]"], label="EnergyTransfer Actual Realtime [kWh]", color = color_actual_realtime, linestyle = linestyle_actual_realtime)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Power [kW]")
plt.title("EnergyTransfer Power")
plt.grid(True)
plt.legend(loc = "upper right")

# グラフの表示と保存
# plt.xticks(rotation=45)  # ラベルの回転
plt.tight_layout()  # レイアウトの自動調整

plt.savefig(f"{folder_path}/stateofsoc_{current_time}.png")
plt.show()

# ----------------------------- 行動変化 --------------------------------
# プロットの作成
plt.figure(figsize=(30, 25))  # グラフのサイズを適切に調整

plt.subplot(4, 1, 1)
plt.plot(df.index, df["natural_action_bid[kWh]"], label="Natural Action Bid [kWh]", color = color_bid, linestyle = linestyle_actual_realtime)
plt.plot(df.index, df["natural_action_realtime[kWh]"], label="Natural Action Realtime [kWh]", color = color_realtime, linestyle = linestyle_actual_realtime)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [kWh]")
plt.title("Natural Action")
plt.grid(True)
plt.legend(loc = "upper right")

plt.subplot(4, 1, 2)
plt.plot(df.index, df["natural_action_bid[kWh]"], label="Natural Action Bid [kWh]", color = color_bid, linestyle = linestyle_actual_realtime)
plt.plot(df.index, df["charge/discharge_bid[kWh]"], label="Charge/Discharge Bid [kWh]", color = color_bid, linestyle = linestyle_bid)
plt.plot(df.index, df["charge/discharge_actual_bid[kWh]"], label="Charge/Discharge Actual Bid [kWh]", color = color_actual_bid, linestyle = linestyle_actual_bid)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [kWh]")
plt.title("Action Bid")
plt.grid(True)
plt.legend(loc = "upper right")

plt.subplot(4, 1, 3)
plt.plot(df.index, df["natural_action_realtime[kWh]"], label="Natural Action Realtime [kWh]", color = color_realtime, linestyle = linestyle_actual_realtime)
plt.plot(df.index, df["charge/discharge_realtime[kWh]"], label="Charge/Discharge Realtime [kWh]", color = color_realtime, linestyle = linestyle_realtime)
plt.plot(df.index, df["charge/discharge_actual_realtime[kWh]"], label="Charge/Discharge Actual Realtime [kWh]", color = color_actual_realtime, linestyle = linestyle_actual_realtime)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [kWh]")
plt.title("Action Realtime")
plt.grid(True)
plt.legend(loc = "upper right")

plt.subplot(4, 1, 4)
plt.plot(df.index, df["charge/discharge_actual_bid[kWh]"], label="Charge/Discharge Actual Bid [kWh]", color = color_actual_bid, linestyle = linestyle_actual_bid)
plt.plot(df.index, df["charge/discharge_actual_realtime[kWh]"], label="Charge/Discharge Actual Realtime [kWh]", color = color_actual_realtime, linestyle = linestyle_actual_realtime)
# 横軸のラベル設定
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとにラベルを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日付のフォーマット指定
plt.xlabel("Date")
plt.ylabel("Value [kWh]")
plt.title("Action Actual")
plt.grid(True)
plt.legend(loc = "upper right")

# グラフの表示と保存
# plt.xticks(rotation=45)  # ラベルの回転
plt.tight_layout()  # レイアウトの自動調整

plt.savefig(f"{folder_path}/editedaction_{current_time}.png")
plt.show()


