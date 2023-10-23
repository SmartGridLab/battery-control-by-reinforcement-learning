import pandas as pd

class ESS_operate:
    def __init__(self, year, month, day, hour):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour

    def operate(self, year, month, day, hour):
        # result_dataframe.csvを読み込む
        dataframe = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

        # 指定した(制御する)日時の部分だけ抽出する
        operate_dataframe = dataframe[(dataframe['year'] == year) &
                  (dataframe['month'] == month) &
                  (dataframe['day'] == day) &
                  (dataframe['hour'] == hour)]
        
        # PVの予測値と実測値の差を計算
        delta_PV = operate_dataframe['PV_actual'] - operate_dataframe['PV_predict']

        # PVが計画よりも多い場合
        if delta_PV >= 0:
            operate_dataframe['energytransfer_bid'] += delta_PV

            ###他のパラメータをそのまま使う
        else :
            operate_dataframe['energytransfer_bid'] -= delta_PV




