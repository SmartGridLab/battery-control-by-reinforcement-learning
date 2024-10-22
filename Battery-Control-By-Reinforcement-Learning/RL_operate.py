import pandas as pd
import parameters

'''
Battery_operationクラス
- 蓄電池を実際に動作させる日の充放電量を決定するmethodを集めたclass。
- input:result_dataframe.csv
- output:result_dataframe.csv

各methodの内容
- operation_bid: 
    実際の充放電を行う時は、PVの発電量が予測とずれる。そのため、売電量やSoCが予測時(energytransfer_bidを計画した時)
    と異なり、計画通りのenergytransfer_bidが実行できない。そのため、実際の動作を可能な限り計画値(energytransfer_bid)に近づけるように動作する。
    そのため、PVの発電量('PV_actual')と予測値('PV_predict_bid')の差を計算し、その差分によって充放電量を調整する。
'''

class Battery_operate():
    def __init__(self):
        # パラメータクラスのインスタンス化
        param = parameters.Parameters() 
        # パラメータの読み込み
        self.BATTERY_CAPACITY = param.BATTERY_CAPACITY
        self.INITIAL_SOC = param.INITIAL_SOC
    
    def get_current_date(self):
        date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        # date_infoは {'year': year, 'month': month, 'day': day} の形式
        date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
        latest_date = date_info['date'].max()
        year = latest_date.year
        month = latest_date.month
        day = latest_date.day
        return year, month, day
   
    def operate_bid(self):
        # 現在の日付を取得
        year, month, day = self.get_current_date()
        df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        # 最新日付のデータをフィルタリング(indexを0から再配置)
        df_result = df_original[(df_original['year'] == year) & 
                                (df_original['month'] == month) & 
                                (df_original['day'] == day)].reset_index(drop = True)
        # PVの予測値('PV_actual[kW]')と実測値('PV_predict_bid[kW]')の差を計算(0.5をかけて[kWh]に変換)
        delta_PV_bid = df_result["PV_actual[kW]"]*0.5 - df_result["PV_predict_bid[kW]"]*0.5
        for j in range(len(df_result)):
            # PVが計画よりも多い場合
            if delta_PV_bid[j] >= 0:
                # caseを記録
                df_result.at[j, 'operation_case'] = 1 #Case1: 充電量増加(放電量抑制)・売電量変化なし

                # 充電量増加(放電量抑制)・売電量変化なし
                df_result.at[j, 'charge/discharge_actual_bid[kWh]'] = df_result.at[j, 'charge/discharge_bid[kWh]'] - abs(delta_PV_bid[j]) #充電量は負の値なので、値を負の方向へ
                df_result.at[j, 'energytransfer_actual_bid[kWh]'] = df_result.at[j, 'energytransfer_bid[kWh]']
            
                ## SoCのチェック
                # SoCの計算
                if j == 0:
                    # INITIAL_SOC = 0.5なので[%]に変換
                    previous_soc = self.INITIAL_SOC *100  ### この実装で良いかは要検討
                else:
                    previous_soc = self.df_result.at[j-1, 'SoC_actual_bid[%]']
                # 定格容量[kWh]で割って[%]変換（charge/discharge_actual_bidは元々[kWh]）
                soc = previous_soc - (self.df_result.at[j, 'charge/discharge_actual_bid[kWh]'])*100/self.BATTERY_CAPACITY

                # SoCが100[%]に到達した場合
                if soc > 100:
                    df_result.at[j, 'mode'] = 3 # Case3: 
                    # 充電できないPV発電量の計算
                    soc_over_enegy = (soc-100)*0.01*self.BATTERY_CAPACITY / 0.5    #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                    # 充電量はSoC100までの量
                    df_result.at[j, 'charge/discharge_actual_bid[kWh]'] += soc_over_enegy #充電量は負の値のため、正方向が減少
                    soc = 100
                    # 差分は売電量を増加させる
                    df_result.at[j, 'energytransfer_actual_bid[kWh]'] += soc_over_enegy
                
                # SoCが0に到達
                if soc < 0:
                    # オーバーした出力
                    soc_over_enegy = (0-soc)*0.01*self.BATTERY_CAPACITY / 0.5 #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                    # 放電量はSoCが0[%]になるまでの量
                    df_result.at[j, 'charge/discharge_actual_bid[kWh]'] -= soc_over_enegy
                    soc = 0
                    # 差分だけ売電量減少
                    df_result.at[j, 'energytransfer_actual_bid[kWh]'] -= soc_over_enegy
                    df_result.at[j, 'mode'] = 5 # Case5:
                        
            # PVが計画よりも少ない場合
            else:
                # caseを記録
                df_result.at[j, 'mode'] = -1

                # 充電量抑制(放電量増加)・売電量変化なし
                df_result.at[j, 'charge/discharge_actual_bid[kWh]'] = df_result.at[j, 'charge/discharge_bid[kWh]'] + abs(delta_PV_bid[j])    #充電量は負の値なので、値を正の方向へ
                df_result.at[j, 'energytransfer_actual_bid[kWh]'] = df_result.at[j, 'energytransfer_bid[kWh]']
            
                # SoCの計算
                if j == 0:
                    previous_soc = self.INITIAL_SOC  # この実装で良いかは要検討
                else:
                    previous_soc = df_result.at[j-1, 'SoC_actual_bid[%]']
                soc = previous_soc - (df_result.at[j, 'charge/discharge_actual_bid[kWh]']*0.5)*100/self.BATTERY_CAPACITY

                # SoCが100に到達した場合
                if soc > 100:
                    # caseを記録
                    df_result.at[j, 'mode'] = -3
                    # オーバーした入力
                    soc_over_enegy = (soc-100)*0.01*self.BATTERY_CAPACITY / 0.5    #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                    # 充電量はSoC100までの量
                    df_result.at[j, 'charge/discharge_actual_bid[kWh]'] += soc_over_enegy #充電量は負の値のため、正方向が減少
                    soc = 100
                    # 差分は売電量を増加させる
                    df_result.at[j, 'energytransfer_actual_bid[kWh]'] += soc_over_enegy

                # if:SoCが0に到達
                if soc < 0:
                    # caseを記録
                    df_result.at[j, 'mode'] = -5
                    # オーバーした出力
                    soc_over_enegy = (0-soc)*0.01*self.BATTERY_CAPACITY / 0.5 #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                    # 放電量はSoC0までの量
                    df_result.at[j, 'charge/discharge_actual_bid[kWh]'] -= soc_over_enegy
                    soc = 0
                    # 差分だけ売電量減少
                    df_result.at[j, 'energytransfer_actual_bid[kWh]'] -= soc_over_enegy

            # energytransfer_actual_bidの修正
            if df_result.at[j, 'energytransfer_actual_bid[kWh]'] < 0:
                df_result.at[j, 'energytransfer_actual_bid[kWh]'] = 0
                ## デバッグ用。energytransfer_actual_bidが負の値になっていたらおかしいので-999を入れておく
                df_result.at[j, 'mode'] = -999

            # 'SoC_actual_bid'へsocを代入
            df_result.at[j, 'SoC_actual_bid[%]'] = soc
        return df_result, df_original

    def operate_realtime(self):
        # 現在日付を取得
        year, month, day = self.get_current_date()
        df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        # 最新日付のデータをフィルタリング(indexを0から再配置)
        df_result = df_original[(df_original['year'] == year) & 
                                (df_original['month'] == month) & 
                                (df_original['day'] == day)].reset_index(drop = True)
        # PVの予測値('PV_actual[kW]')と実測値('PV_predict_realtime[kW]')の差を計算
        delta_PV_realtime = df_result["PV_actual[kW]"] - df_result["PV_predict_realtime[kW]"]
        for j in range(len(df_result)):
            # PVが計画よりも多い場合
            if delta_PV_realtime[j] >= 0:
                # caseを記録
                df_result.at[j, 'operation_case'] = 1 #Case1: 充電量増加(放電量抑制)・売電量変化なし

                # 充電量増加(放電量抑制)・売電量変化なし
                df_result.at[j, 'charge/discharge_actual_realtime[kWh]'] = df_result.at[j, 'charge/discharge_realtime[kWh]'] - abs(delta_PV_realtime[j]) #充電量は負の値なので、値を負の方向へ
                df_result.at[j, 'energytransfer_actual_realtime[kWh]'] = df_result.at[j, 'energytransfer_realtime[kWh]']
            
                ## SoCのチェック
                # SoCの計算
                if j == 0:
                    previous_soc = self.INITIAL_SOC ### この実装で良いかは要検討
                else:
                    previous_soc = df_result.at[j-1, 'SoC_actual_realtime[%]']
                # 出力[kW]を30分あたりの電力量[kWh]に変換、定格容量[kWh]で割って[%]変換
                soc = previous_soc - (df_result.at[j, 'charge/discharge_actual_realtime[kWh]']*0.5)*100/self.BATTERY_CAPACITY

                # SoCが100[%]に到達した場合
                if soc > 100:
                    df_result.at[j, 'mode'] = 3 # Case3: 
                    # 充電できないPV発電量の計算
                    soc_over_enegy = (soc-100)*0.01*self.BATTERY_CAPACITY / 0.5    #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                    # 充電量はSoC100までの量
                    df_result.at[j, 'charge/discharge_actual_realtime[kWh]'] += soc_over_enegy #充電量は負の値のため、正方向が減少
                    soc = 100
                    # 差分は売電量を増加させる
                    df_result.at[j, 'energytransfer_actual_realtime[kWh]'] += soc_over_enegy
                
                # SoCが0に到達
                if soc < 0:
                    # オーバーした出力
                    soc_over_enegy = (0-soc)*0.01*self.BATTERY_CAPACITY / 0.5 #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                    # 放電量はSoCが0[%]になるまでの量
                    df_result.at[j, 'charge/discharge_actual_realtime[kWh]'] -= soc_over_enegy
                    soc = 0
                    # 差分だけ売電量減少
                    df_result.at[j, 'energytransfer_actual_realtime[kWh]'] -= soc_over_enegy
                    df_result.at[j, 'mode'] = 5 # Case5:
                        
            # PVが計画よりも少ない場合
            else:
                # caseを記録
                df_result.at[j, 'mode'] = -1

                # 充電量抑制(放電量増加)・売電量変化なし
                df_result.at[j, 'charge/discharge_actual_realtime[kWh]'] = df_result.at[j, 'charge/discharge_realtime[kWh]'] + abs(delta_PV_realtime[j])    #充電量は負の値なので、値を正の方向へ
                df_result.at[j, 'energytransfer_actual_realtime[kWh]'] = df_result.at[j, 'energytransfer_realtime[kWh]']
            
                # SoCの計算
                if j == 0:
                    # INITIAL_SOC = 0.5なので[%]に変換
                    previous_soc = self.INITIAL_SOC *100  # この実装で良いかは要検討
                else:
                    previous_soc = df_result.at[j-1, 'SoC_actual_realtime[%]']
                soc = previous_soc - (df_result.at[j, 'charge/discharge_actual_realtime[kWh]']*0.5)*100/self.BATTERY_CAPACITY

                # SoCが100に到達した場合
                if soc > 100:
                    # caseを記録
                    df_result.at[j, 'mode'] = -3
                    # オーバーした入力
                    soc_over_enegy = (soc-100)*0.01*self.BATTERY_CAPACITY / 0.5    #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                    # 充電量はSoC100までの量
                    df_result.at[j, 'charge/discharge_actual_realtime[kWh]'] += soc_over_enegy #充電量は負の値のため、正方向が減少
                    soc = 100
                    # 差分は売電量を増加させる
                    df_result.at[j, 'energytransfer_actual_realtime[kWh]'] += soc_over_enegy

                # if:SoCが0に到達
                if soc < 0:
                    # caseを記録
                    df_result.at[j, 'mode'] = -5
                    # オーバーした出力
                    soc_over_enegy = (0-soc)*0.01*self.BATTERY_CAPACITY / 0.5 #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
                    # 放電量はSoC0までの量
                    df_result.at[j, 'charge/discharge_actual_realtime[kWh]'] -= soc_over_enegy
                    soc = 0
                    # 差分だけ売電量減少
                    df_result.at[j, 'energytransfer_actual_realtime[kWh]'] -= soc_over_enegy

            # energytransfer_actual_bidの修正
            if df_result.at[j, 'energytransfer_actual_realtime[kWh]'] < 0:
                df_result.at[j, 'energytransfer_actual_realtime[kWh]'] = 0
                ## デバッグ用。energytransfer_actual_bidが負の値になっていたらおかしいので-999を入れておく
                df_result.at[j, 'mode'] = -999

            # 'SoC_actual_bid'へsocを代入
            df_result.at[j, 'SoC_actual_realtime[%]'] = soc
        return df_result, df_original

    def mode_dependent_operate(self, mode):
        if mode == "bid":
            df_result, df_original = self.operate_bid()
        elif mode == "realtime":
            df_result, df_original = self.operate_realtime()
        # year, month, day, hourをindexとして設定
        df_result.set_index(['year', 'month', 'day', 'hour'], inplace = True)
        df_original.set_index(['year', 'month', 'day', 'hour'], inplace = True)
        # 該当日付を更新
        df_original.update(df_result)
        # indexを振り直す
        df_original.reset_index(inplace = True)
        # df_resultをdf_result.csvへ上書き保存
        df_original.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", header = True, index=False)