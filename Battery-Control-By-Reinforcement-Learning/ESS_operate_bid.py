import pandas as pd

# 蓄電池容量
battery_MAX = 4

# result_dataframe.csvを読み込む
df_result = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

print("-bidモード機器動作開始-")

# PVの予測値('PV_actual')と実測値('PV_predict_bid')の差を計算
delta_PV = df_result["PV_actual"] - df_result["PV_predict_bid"]

for i in range(len(df_result):

    # PVが計画よりも多い場合
    if delta_PV >= 0:
        # 充電量増加(放電量抑制)・売電量変化なし
        dataframe.at[j, 'charge/discharge_actual_bid'] = dataframe.at[j, 'charge/discharge_bid'] - abs(delta_PV) #充電量は負の値なので、値を負の方向へ
        dataframe.at[j, 'energytransfer_actual_bid'] = dataframe.at[j, 'energytransfer_bid']

        dataframe.at[j, 'mode'] = 1
    
        ## SoCのチェック
        # SoCの計算
        if j == 0:
            before_soc = 0 ###マジックナンバー
        else:
            before_soc = dataframe.at[j, 'SoC_actual_bid']
        soc = before_soc - (dataframe.at[j, 'charge/discharge_actual_bid']*0.5)*100/battery_MAX   #出力[kW]を30分あたりの電力量[kWh]に変換、定格容量[kWh]で割って[%]変換
        
        # SoCが100に到達した場合
        if soc > 100:
            # オーバーした入力
            soc_over_enegy = (soc-100)*0.01*battery_MAX / 0.5    #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
            # 充電量はSoC100までの量
            dataframe.at[j, 'charge/discharge_actual_bid'] += soc_over_enegy #充電量は負の値のため、正方向が減少
            soc = 100
            # 差分は売電量を増加させる
            dataframe.at[j, 'energytransfer_actual_bid'] += soc_over_enegy

            dataframe.at[j, 'mode'] = 3
        
        # SoCが0に到達
        if soc < 0:
            # オーバーした出力
            soc_over_enegy = (0-soc)*0.01*battery_MAX / 0.5 #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
            # 放電量はSoC0までの量
            dataframe.at[j, 'charge/discharge_actual_bid'] -= soc_over_enegy
            soc = 0
            # 差分だけ売電量減少
            dataframe.at[j, 'energytransfer_actual_bid'] -= soc_over_enegy

            dataframe.at[j, 'mode'] = 5
                

    # PVが計画よりも少ない場合
    else:
        # 充電量抑制(放電量増加)・売電量変化なし
        dataframe.at[j, 'charge/discharge_actual_bid'] = dataframe.at[j, 'charge/discharge_bid'] + abs(delta_PV)    #充電量は負の値なので、値を正の方向へ
        dataframe.at[j, 'energytransfer_actual_bid'] = dataframe.at[j, 'energytransfer_bid']

        dataframe.at[j, 'mode'] = -1
    
        # SoCの計算
        if j == 0:
            before_soc = 0 ###マジックナンバー
        else:
            before_soc = dataframe.at[j, 'SoC_actual_bid']
        soc = before_soc - (dataframe.at[j, 'charge/discharge_actual_bid']*0.5)*100/battery_MAX

        # SoCが100に到達した場合
        if soc > 100:
            # オーバーした入力
            soc_over_enegy = (soc-100)*0.01*battery_MAX / 0.5    #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
            # 充電量はSoC100までの量
            dataframe.at[j, 'charge/discharge_actual_bid'] += soc_over_enegy #充電量は負の値のため、正方向が減少
            soc = 100
            # 差分は売電量を増加させる
            dataframe.at[j, 'energytransfer_actual_bid'] += soc_over_enegy

            dataframe.at[j, 'mode'] = -3

        # if:SoCが0に到達
        if soc < 0:
            # オーバーした出力
            soc_over_enegy = (0-soc)*0.01*battery_MAX / 0.5 #オーバーしたSoC[%] -> 30分あたりの電力量[kWh] -> 出力[kW]
            # 放電量はSoC0までの量
            dataframe.at[j, 'charge/discharge_actual_bid'] -= soc_over_enegy
            soc = 0
            # 差分だけ売電量減少
            dataframe.at[j, 'energytransfer_actual_bid'] -= soc_over_enegy

            dataframe.at[j, 'mode'] = -5

    # energytransfer_actual_bidの修正
    if dataframe.at[j, 'energytransfer_actual_bid'] < 0:
        dataframe.at[j, 'energytransfer_actual_bid'] = 0

        dataframe.at[j, 'mode'] *= 2

    # SoCの最終処理
    if j == 0:
        dataframe.at[j, 'SoC_actual_bid'] = before_soc
    if j < last_csv_row:
        dataframe.at[j+1, 'SoC_actual_bid'] = soc  #最終行以降にsoc格納するスペースはない
    
    # サイクル
    i += 1
###

# result_dataframe.csvを上書き保存
dataframe.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)

print("-bidモード機器動作終了-")