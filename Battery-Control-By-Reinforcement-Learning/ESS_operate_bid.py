import pandas as pd
import parameters as Param


def __init__(self):
    # パラメータクラスのインスタンス化
    param = Param() 
    # パラメータの読み込み
    self.BATTERY_CAPACITY = param.BATTERY_CAPACITY
    self.INITIAL_SOC = param.INITIAL_SOC

    # RLの結果を書き込んだファイル(result_self.df_result.csv)を読み込む
    self.df_result = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_self.df_result.csv")

    # PVの予測値('PV_actual')と実測値('PV_predict_bid')の差を計算
    self.delta_PV = self.df_result["PV_actual"] - self.df_result["PV_predict_bid"]

def battery_operation(self):
    for j in range(len(self.df_result)):
        # PVが計画よりも多い場合
        if self.delta_PV >= 0:
            # caseを記録
            self.df_result.at[j, 'operation_case'] = 1 #Case1: 充電量増加(放電量抑制)・売電量変化なし

            # 充電量増加(放電量抑制)・売電量変化なし
            self.df_result.at[j, 'charge/discharge_actual_bid'] = self.df_result.at[j, 'charge/discharge_bid'] - abs(self.delta_PV) #充電量は負の値なので、値を負の方向へ
            self.df_result.at[j, 'energytransfer_actual_bid'] = self.df_result.at[j, 'energytransfer_bid']
        
            ## SoCのチェック
            # SoCの計算
            if j == 0:
                previous_soc = self.INITIAL_SOC ### この実装で良いかは要検討
            else:
                previous_soc = self.df_result.at[j, 'SoC_actual_bid']
            # 出力[kW]を30分あたりの電力量[kWh]に変換、定格容量[kWh]で割って[%]変換
            soc = previous_soc - (self.df_result.at[j, 'charge/discharge_actual_bid']*0.5)*100/self.BATTERY_CAPACITY
            
            # SoCが100[%]に到達した場合
            if soc > 100:
                self.df_result.at[j, 'mode'] = 3 # Case3: 
                # 充電できないPV発電量の計算
                soc_over_enegy = (soc-100)*0.01*self.BATTERY_CAPACITY / 0.5    #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                # 充電量はSoC100までの量
                self.df_result.at[j, 'charge/discharge_actual_bid'] += soc_over_enegy #充電量は負の値のため、正方向が減少
                soc = 100
                # 差分は売電量を増加させる
                self.df_result.at[j, 'energytransfer_actual_bid'] += soc_over_enegy
            
            # SoCが0に到達
            if soc < 0:
                # オーバーした出力
                soc_over_enegy = (0-soc)*0.01*self.BATTERY_CAPACITY / 0.5 #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                # 放電量はSoCが0[%]になるまでの量
                self.df_result.at[j, 'charge/discharge_actual_bid'] -= soc_over_enegy
                soc = 0
                # 差分だけ売電量減少
                self.df_result.at[j, 'energytransfer_actual_bid'] -= soc_over_enegy
                self.df_result.at[j, 'mode'] = 5 # Case5:
                    

        # PVが計画よりも少ない場合
        else:
            # caseを記録
            self.df_result.at[j, 'mode'] = -1

            # 充電量抑制(放電量増加)・売電量変化なし
            self.df_result.at[j, 'charge/discharge_actual_bid'] = self.df_result.at[j, 'charge/discharge_bid'] + abs(self.delta_PV)    #充電量は負の値なので、値を正の方向へ
            self.df_result.at[j, 'energytransfer_actual_bid'] = self.df_result.at[j, 'energytransfer_bid']
        
            # SoCの計算
            if j == 0:
                previous_soc = self.INITIAL_SOC  # この実装で良いかは要検討
            else:
                previous_soc = self.df_result.at[j, 'SoC_actual_bid']
            soc = previous_soc - (self.df_result.at[j, 'charge/discharge_actual_bid']*0.5)*100/self.BATTERY_CAPACITY

            # SoCが100に到達した場合
            if soc > 100:
                # caseを記録
                self.df_result.at[j, 'mode'] = -3
                # オーバーした入力
                soc_over_enegy = (soc-100)*0.01*self.BATTERY_CAPACITY / 0.5    #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                # 充電量はSoC100までの量
                self.df_result.at[j, 'charge/discharge_actual_bid'] += soc_over_enegy #充電量は負の値のため、正方向が減少
                soc = 100
                # 差分は売電量を増加させる
                self.df_result.at[j, 'energytransfer_actual_bid'] += soc_over_enegy

            # if:SoCが0に到達
            if soc < 0:
                # caseを記録
                self.df_result.at[j, 'mode'] = -5
                # オーバーした出力
                soc_over_enegy = (0-soc)*0.01*self.BATTERY_CAPACITY / 0.5 #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                # 放電量はSoC0までの量
                self.df_result.at[j, 'charge/discharge_actual_bid'] -= soc_over_enegy
                soc = 0
                # 差分だけ売電量減少
                self.df_result.at[j, 'energytransfer_actual_bid'] -= soc_over_enegy

        # energytransfer_actual_bidの修正
        if self.df_result.at[j, 'energytransfer_actual_bid'] < 0:
            self.df_result.at[j, 'energytransfer_actual_bid'] = 0
            ## デバッグ用。energytransfer_actual_bidが負の値になっていたらおかしいので-999を入れておく
            self.df_result.at[j, 'mode'] = -999


    # result_self.df_result.csvを上書き保存
    self.df_result.to_csv("Battery-Control-By-Reinforcement-Learning/result_self.df_result.csv", index=False)

    print("-bidモード機器動作終了-")