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

        # 蓄電池容量
        battery_MAX = 4

        # 指定した(制御する)日時の部分だけ抽出する
        operate_dataframe = dataframe[(dataframe['year'] == year) &
                  (dataframe['month'] == month) &
                  (dataframe['day'] == day) &
                  (dataframe['hour'] == hour)]
        
        # PVの予測値と実測値の差を計算
        delta_PV = operate_dataframe['PV_actual'] - operate_dataframe['PV_predict']

        # PVが計画よりも多い場合
        if delta_PV >= 0:
            # 蓄電池動作なし、充電中の場合
            if operate_dataframe['charge/discharge_bid'] <= 0:
                # 充電量増加、売電量変化なし
                operate_dataframe['charge/discharge_actual'] - delta_PV #充電量は負の値
                operate_dataframe['energytransfer_actual'] = operate_dataframe['energytransfer_bid']
                #operate_dataframe['SoC_actual'] = operate_dataframe['SoC_bid'] + (delta_PV * 0.5)/battery_MAX   #出力[kW]を30分あたりの電力量[kWh]に変換、定格容量[kWh]で割って[%]変換
            
                # SoCが100に到達した場合
                if operate_dataframe['SoC_actual'] > 100:
                    soc_over_enegy = (operate_dataframe['SoC_actual'] - 100) * battery_MAX / 0.5    #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                    # 充電量はSoC100までの量
                    operate_dataframe['charge/discharge_actual'] + soc_over_enegy #充電量は負の値のため、正方向が減少
                    operate_dataframe['SoC_actual'] = 100
                    # のこりは売電量に回す
                    operate_dataframe['energytransfer_actual'] += soc_over_enegy


            # 放電時の場合
            else :
                # delta_PV<放電出力の場合
                if delta_PV < operate_dataframe['energytransfer_bid']:
                    #放電量抑制、売電量変化なし
                    operate_dataframe['energytransfer_actual'] = operate_dataframe['energytransfer_bid']
                    operate_dataframe['charge/discharge_actual'] -= delta_PV
                    operate_dataframe['SoC_actual'] = operate_dataframe['SoC_bid'] - (delta_PV * 0.5)/battery_MAX

                # if:delta_PV>放電出力
                else:
                    # 充電に切り替え、売電量変化なし
                    operate_dataframe['energytransfer_actual'] = operate_dataframe['energytransfer_bid']
                    operate_dataframe['charge/discharge_actual'] -= delta_PV
                    operate_dataframe['SoC_actual'] = operate_dataframe['SoC_bid'] - (delta_PV * 0.5)/battery_MAX

                    # if:SoCが100に到達
                        # 充電量はSoC100までの量
                        # のこりは売電量に回す
        # PVが計画よりも多い場合
        else :
            print("エラー除けのダミー")
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




