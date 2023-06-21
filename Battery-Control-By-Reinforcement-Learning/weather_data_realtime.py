#import math
import urllib.request

import datetime
import pygrib
import pandas as pd
import numpy as np

print("---リアルタイム制御用気象予報データ抽出プログラム開始---\n")

#UTCとJSTの時差
time_diff = datetime.timedelta(hours=9) 

#UTC時刻で入力
today = datetime.date.today() + time_diff
now = datetime.datetime.now() + time_diff
current_time = now.strftime("%H")
current_time = int(current_time)

current_time = 18   #仮入力開発環境時に使用

#データパス設定
#JSTとUTCの日付が異なるとき(データ利用時間を考慮して0000-1130)
if current_time >= 0 and current_time <12:  
    data_year = (today - datetime.timedelta(days=1)).strftime("%Y")
    data_date = (today - datetime.timedelta(days=1)).strftime("%m%d")
    data_date1 = (today - datetime.timedelta(days=1)).strftime("%Y/%m/%d")

    #0000-0230
    if current_time < 3:
        data_time = "120000"
    #0300-0530
    elif current_time < 6:
        data_time = "150000"
    #0600-0830
    elif current_time < 9:
        data_time = "180000"
    #0900-1130
    elif current_time < 12:
        data_time = "210000"

#JSTとUTCが同じ日付になるとき
else:
    data_year = today.strftime("%Y")
    data_date = today.strftime("%m%d")
    data_date1 = today.strftime("%Y/%m/%d")

    #1200-1430
    if current_time < 15:
        data_time = "000000"
    #1500-1730
    elif current_time < 18:
        data_time = "030000"
    #1800-2030
    elif current_time < 21:
        data_time = "060000"
    #1800-2030
    elif current_time < 24:
        data_time = "090000"
    

#動作確認用
data_year = 2023    #仮入力開発環境時に使用
data_date = "0131"    #仮入力開発環境時に使用

#緯度指定
lat =36.06489716079195
lon = 140.1349848817127

#緯度 0.05度刻み
lat1 = lat - 0.025
lat2 = lat + 0.025
#経度 0.0625度刻み
lon1 = lon - 0.03125
lon2 = lon + 0.03125

print("緯度 : " + str(lat))
print("経度 : " + str(lon) + "\n")

print("今日の日付:" + str(today.strftime("%Y/%m/%d")))
print("現在時刻:" + str(now.strftime("%H:%M") + "\n"))

#print("今日の日付:" + data_date)   #仮入力開発環境時に使用
#print("現在時刻:" + str(current_time + "\n")) #仮入力開発環境時に使用
print(str(data_date1) + " " + data_time + "(UTC)公開の予測データを取得\n")    #ローカル環境時に使用



#---------------------------------------------------------------------------------------------------------
#パラメータメモ
#prmsl = gpv_file.select(parameterName='Pressure reduced to MSL')            #[0] 海面更正気圧[Pa]
#sp    = gpv_file.select(parameterName='Pressure')                           #[1] 気圧[Pa]
#uwind = gpv_file.select(parameterName='u-component of wind')                #[2] 風速(東西)[m/s]
#vwind = gpv_file.select(parameterName='v-component of wind')                #[3] 風速(南北)[m/s]
#temp  = gpv_file.select(parameterName='Temperature')                        #[4] 気温[K]
#rh    = gpv_file.select(parameterName='Relative humidity')                  #[5] 相対湿度[%]
#lcc   = gpv_file.select(parameterName='Low cloud cover')                    #[6] 下層雲量[%]
#mcc   = gpv_file.select(parameterName='Medium cloud cover')                 #[7] 中層雲量[%]
#hcc   = gpv_file.select(parameterName='High cloud cover')                   #[8] 上層雲量[%]
#tcc   = gpv_file.select(parameterName='Total cloud cover')                  #[9] 全雲量[%]
#tp    = gpv_file.select(parameterName='Total precipitation')                #[10] 降水量[kg/m^2]
#dswrf = gpv_file.select(parameterName='Downward short-wave radiation flux') #[11] 下向き短波放射フラックス[W/m^2]
#---------------------------------------------------------------------------------------------------------

#最終データの型枠
df = pd.DataFrame(columns=["year","month","day","hour","Pressure","temperature","u-component of wind","v-component of wind","Relative humidity", "Total cloud cover", "Total precipitation"])

#関数：データ取得
def data_acquisition(data_year, data_date, data_time, data_range):

    ## GRIB2ファイルを読み込む
    dataname_base = "Battery-Control-By-Reinforcement-Learning/"
    dataname_base1 = "Z__C_RJTD_"
    dataname_base2 = "_MSM_GPV_Rjp_Lsurf_FH"
    dataname_base3 = "_grib2.bin"

    #ファイル名
    DL_file_name = dataname_base1 + str(data_year) + str(data_date) + data_time + dataname_base2 + data_range + dataname_base3
    file_name = dataname_base + DL_file_name


    #ファイルダウンロード
    #print(data_range +"時間後予測  ダウンロード開始...")    #ローカル環境時に使用
    #url_surf = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/" + str(data_date1) + "/" + DL_file_name     ##ローカル環境時に使用
    #urllib.request.urlretrieve(url_surf, DL_file_name)    #ローカル環境時に使用
    #print(data_range +"時間後予測  ダウンロード完了")


    #ファイルオープン
    gpv_file = pygrib.open(file_name)   #開発環境時に使用
    #gpv_file = pygrib.open(DL_file_name) #ローカル環境時に使用
    print(data_range +"時間後予測  取得開始...")
    

    #ファイル抽出
    p_messages  = gpv_file.select(parameterName='Pressure')
    t_messages = gpv_file.select(parameterName='Temperature')
    uw_messages = gpv_file.select(parameterName='u-component of wind')
    vw_messages = gpv_file.select(parameterName='v-component of wind')
    rh_messages  = gpv_file.select(parameterName='Relative humidity')
    tcc_messages  = gpv_file.select(parameterName='Total cloud cover')
    tp_messages  = gpv_file.select(parameterName='Total precipitation')
    dswrf_messages  = gpv_file.select(parameterName='Downward short-wave radiation flux')

    #時系列取り出し・データ分解
    df_validdata_ = pd.DataFrame({"validDate": [msg.validDate + time_diff for msg in t_messages]})
    df_validdata = pd.DataFrame(columns=["year","month","day","hour"])
    df_validdata['year'] = df_validdata_['validDate'].dt.year
    df_validdata['month'] = df_validdata_['validDate'].dt.month
    df_validdata['day'] = df_validdata_['validDate'].dt.day
    df_validdata['hour'] = df_validdata_['validDate'].dt.hour

    #データ格納
    df1 = pd.DataFrame({
        "Pressure":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] * 0.01 for msg in p_messages
        ]
   })
    df2 = pd.DataFrame({
        "temperature": [
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] - 273.15 for msg in t_messages
        ]
    })
    df3 = pd.DataFrame({
        "u-component of wind":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in uw_messages
        ]
    })
    df4 = pd.DataFrame({
        "v-component of wind":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in vw_messages
        ]
    })
    df5 = pd.DataFrame({
        "Relative humidity":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in rh_messages
        ]
    })
    df6 = pd.DataFrame({
        "Total cloud cover":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in tcc_messages
        ]
    })
    df7 = pd.DataFrame({
        "Total precipitation":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in tp_messages
        ]
    })
    df8 = pd.DataFrame({
        "radiation flux":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in dswrf_messages
        ]
    })

    #データ統合
    df_ = pd.concat([df_validdata, df1], axis=1)
    df_ = pd.concat([df_, df2], axis=1)
    df_ = pd.concat([df_, df3], axis=1)
    df_ = pd.concat([df_, df4], axis=1)
    df_ = pd.concat([df_, df5], axis=1)
    df_ = pd.concat([df_, df6], axis=1)
    df_ = pd.concat([df_, df7], axis=1)
    df_ = pd.concat([df_, df8], axis=1)
    df_.fillna(0)

    print(data_range +"時間後予測  取得完了\n")

    return df_

##0-15時間後データ
df_ = data_acquisition(data_year, data_date, data_time, data_range = "00-15")
df_T = df_.T    #転置
list = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29)  #空の列挿入(毎時30分用)
for i in list:  
    df_T.insert(i, i + 0.5, np.nan)
df_ = df_T.T    #転置
df = pd.concat([df, df_], axis=0)


##16-33時間後(16-27時間後)データ
df_ = data_acquisition(data_year, data_date, data_time, data_range = "16-33")
df_.drop(range(12, 17),inplace=True)  #28-33時間後を削除
df_T = df_.T    #転置
list = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)  #空の列挿入(毎時30分用)
for i in list:
    df_T.insert(i, i + 100.5, np.nan)
df_ = df_T.T    #転置
df = pd.concat([df, df_], axis=0)

#index整理・線形補間
df = df.reset_index(drop=True)
df = df.interpolate()


##最終調整
#年・日付・月をまだぐときの値の処理

for i in range(5,53):
    #年またぎ
    if df.at[i, 'year'] != df.at[i+1, 'year']:
        df.at[i, 'year'] = df.at[i-1, 'year']

    #月またぎ
    if df.at[i, 'month'] != df.at[i+1, 'month']:
        df.at[i, 'month'] = df.at[i-1, 'month']
        df.at[i, 'day'] = df.at[i-1, 'day']

    #日付またぎ
    if df.at[i-1, 'hour'] ==23.0:
        df.at[i, 'hour'] = 23.5
    

#int化
df['year'] = df['year'].astype('int')
df['month'] = df['month'].astype('int')
df['day'] = df['day'].astype('int')

#27時間後を削除
df = df.reset_index(drop=True)
df.drop(54,inplace=True)   

#出力
df.to_csv('Battery-Control-By-Reinforcement-Learning/weather_data_realtime.csv')    #開発環境時に使用
print("--結果出力完了--")
#print(df)
print("\n\n---気象予報データ抽出プログラム終了---")
