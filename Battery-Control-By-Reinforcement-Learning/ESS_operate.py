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
            # if:蓄電池動作なし
                # 充電量増加(売電量変化なし)
                
                # if:SoCが100に到達
                    # 充電量はSoC100までの量
                    # のこりは売電量に回す

            # if:充電時
                # 充電量増加(売電量変化なし)
                
                # if:SoCが100に到達
                    # 充電量はSoC100までの量
                    # のこりは売電量に回す

            # if:放電時
                # if:delta_PV<放電出力
                    #放電量抑制(売電量変化なし)
                # if:delta_PV>放電出力
                    # 充電になる(売電量変化なし)

                    # if:SoCが100に到達
                        # 充電量はSoC100までの量
                        # のこりは売電量に回す

            operate_dataframe['energytransfer_bid'] += delta_PV

        # PVが計画よりも多い場合
        else :
            # if:蓄電池動作なし
                # 放電量増加(売電量変化なし)
                
                # if:SoCが0に到達
                    # 放電量はSoC0までの量
                    # 差分だけ売電量減少

            # if:充電時
                # if:delta_PV<放電出力
                    # 充電量抑制(売電量変化なし)
                
                # if:delta_PV>放電出力
                    # 放電になる(売電量変化なし)

                    # if:SoCが0に到達
                        # 放電量はSoC0までの量
                        # 差分だけ売電量減少

            # if:放電時
                # 放電量増加(売電量変化なし)
                
                # if:SoCが0に到達
                    # 放電量はSoC0までの量
                    # 差分だけ売電量減少
            operate_dataframe['energytransfer_bid'] -= delta_PV




