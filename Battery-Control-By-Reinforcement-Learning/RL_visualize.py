# グラフを描写するための関数を定義

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

def __init__(self):
    self.steps = steps  # envからもらってくる
    self.all_rewards = all_rewards # envからもらってくる

if __name__ == "__main__" :
    pp = PdfPages(pdf_name) # PDFの作成
    graph_1 = self.descr_reward(self.steps, self.all_rewards)
    graph_2 = self.descr_schedule(self.all_action_real, self.all_PV_out_time, self.all_energy_transfer, self.all_soc, mode = 0)
    graph_3 = self.descr_schedule(self.all_action_real, self.all_PV_out_time, self.all_energy_transfer, self.all_soc, mode = 1)

    pp.savefig(graph_1)
    pp.savefig(graph_2)
    pp.savefig(graph_3)

    pp.close()

# 充放電計画のグラフの描写
def descr_schedule(self, action, PVout, energy_transfer, soc, mode):     #修正後のactionを引き渡す
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
        ax1.set_ylabel("Power [kW] / Power rates [Yen]", fontsize=35)
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

# rewardのグラフの描写
def descr_reward(self, steps, reward):
    fig = plt.figure(figsize=(24, 14), dpi=80)
    plt.plot(np.arange(steps), reward, label = "Reward")
    plt.legend(prop={"size": 35})
    plt.xlabel("Episode", fontsize = 35)
    plt.ylabel("Reward", fontsize = 35)
    plt.tick_params(labelsize=35)
    plt.close()
    
    return fig
