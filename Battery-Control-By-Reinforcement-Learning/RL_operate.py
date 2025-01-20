from calendar import month
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
        self.df_test = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        self.boundary_soc_df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/for_debug/boundary_soc.csv")
    
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
        # PVの実測値('PV_actual[kW]')と予測値('PV_predict_bid[kW]')の差を計算(0.5をかけて[kWh]に変換)
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
                    previous_soc = df_result.at[j-1, 'SoC_actual_bid[%]']
                # 定格容量[kWh]で割って[%]変換（charge/discharge_actual_bidは元々[kWh]）
                soc = previous_soc - (df_result.at[j, 'charge/discharge_actual_bid[kWh]'])*100/self.BATTERY_CAPACITY

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

    def operate_plan(self, PV_predict, action_fromRL, current_soc):
        '''
        引数
        - PV_predict: PV発電量の予測値[kW], RL_env.pyではPV実測値を使っている
        - action_fromRL: RLからのアクション, -2.0 ~ 2.0[kW](RL_env.pyで設定)
        - current_soc: 現在のSoC, 0.0 ~ 1.0[割合]
        出力
        - edited_action: RLからのアクションを実際に動作させるために編集した値 = -2.0 ~ 2.0[kWh]
        - next_soc: 次のSoC, 0.0 ~ 1.0[割合]
        - energytransfer: 電力取引量[kWh]
        '''

        current_soc = current_soc * self.BATTERY_CAPACITY # [割合]を[kWh]に変換
        # 充電時
        if action_fromRL < 0:
            if PV_predict + action_fromRL < 0: # PV発電よりも充電計画値が多い
                edited_action = - PV_predict
                _next_soc = current_soc - edited_action
                # 過剰充電
                if _next_soc > 4.0:
                    next_soc = 4.0
                    edited_action = current_soc - next_soc
                # 正常充電
                else:
                    next_soc = _next_soc
            else: # PV予測値内で充電
                edited_action = action_fromRL
                _next_soc = current_soc - edited_action
                # 過剰充電
                if _next_soc > 4.0:
                    next_soc = 4.0
                    edited_action = current_soc - next_soc
                # 正常充電
                else:
                    next_soc = _next_soc
        # 放電時
        else: # action_fromRL >= 0
            # PV発電量予測値は閾値として考慮しない
            edited_action = action_fromRL
            _next_soc = current_soc - edited_action
            # 過剰放電
            if _next_soc < 0.0:
                next_soc = 0.0
                edited_action = current_soc - next_soc
            # 正常放電
            else:
                next_soc = _next_soc
        
        energytransfer = PV_predict + edited_action
        next_soc = next_soc / self.BATTERY_CAPACITY # [kW] -> [割合]

        return edited_action, next_soc, energytransfer
    
    def operate_actual(self, mode):
        soc_list = [self.boundary_soc_df[f"Initial_SoC_actual_{mode}"][0]]
        year, month, day = self.get_current_date()
        df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        df_result = df_original[(df_original['year'] == year) & 
                                (df_original['month'] == month) & 
                                (df_original['day'] == day)].reset_index(drop = True)
        # 実績値を計算する
        for j in range(len(df_result)):
            edited_action = df_result.at[j, f"charge/discharge_{mode}[kWh]"]
            PV_actual = df_result.at[j, "PV_actual[kW]"]
            current_soc = soc_list[-1] * self.BATTERY_CAPACITY # 0.0 ~ 1.0[割合]を0.0 ~ 4.0 [kW]
            # 充電時
            if edited_action < 0:
                # PV発電実測値よりも充電計画値が大きい
                if PV_actual + edited_action < 0:
                    edited_action_actual = - PV_actual
                    _next_soc = current_soc - edited_action_actual
                    # 過剰充電
                    if _next_soc > 4.0:
                        next_soc = 4.0
                        edited_action_actual = current_soc - next_soc
                    # 正常充電
                    else:
                        next_soc = _next_soc
                # PV実測値内で充電
                else:
                    edited_action_actual = edited_action
                    _next_soc = current_soc - edited_action_actual
                    # 過剰充電
                    if _next_soc > 4.0:
                        next_soc = 4.0
                        edited_action_actual = current_soc - next_soc
                    # 正常充電
                    else:
                        next_soc = _next_soc
            # 放電時
            else: # edited_action >= 0
                # PV発電量実測値は閾値として考慮しない
                edited_action_actual = edited_action
                _next_soc = current_soc - edited_action_actual
                # 過剰放電
                if _next_soc < 0.0:
                    next_soc = 0.0
                    edited_action_actual = current_soc - next_soc
                # 正常放電
                else:
                    next_soc = _next_soc

            # 電力取引実績値の計算
            energytransfer_actual = PV_actual + edited_action_actual
            next_soc = next_soc / self.BATTERY_CAPACITY # [kW] -> [割合]

            soc_list.append(next_soc)
            df_result.at[j, f"SoC_actual_{mode}[%]"] = next_soc * 100 # [割合]->[%]
            df_result.at[j, f"charge/discharge_actual_{mode}[kWh]"] = edited_action_actual
            df_result.at[j, f"energytransfer_actual_{mode}[kWh]"] = energytransfer_actual
        
        # 次の日の初期SoC = 前日のSoC終値
        self.boundary_soc_df[f"Initial_SoC_actual_{mode}"][0] = soc_list[-1]
        # 更新
        self.boundary_soc_df.to_csv("Battery-Control-By-Reinforcement-Learning/for_debug/boundary_soc.csv", index = False)

        return df_result, df_original
    
    def mode_dependent_operate(self, mode):
        # 実績値を計算
        df_result, df_original = self.operate_actual(mode)
        # year, month, day, hourをindexとして設定
        df_result.set_index(['year', 'month', 'day', 'hour'], inplace = True)
        df_original.set_index(['year', 'month', 'day', 'hour'], inplace = True)
        # 該当日付を更新
        df_original.update(df_result)
        # indexを振り直す
        df_original.reset_index(inplace = True)
        # df_resultをdf_result.csvへ上書き保存
        df_original.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", header = True, index=False)