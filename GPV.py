import os
import math
import subprocess

import datetime
import pygrib
import pandas as pd
import numpy as np

## GRIB2ファイルを読み込む
#dataname_base1 = "Z__C_RJTD_"
#dataname_base2 = "_MSM_GPV_Rjp_Lsurf_FH"
#dataname_base3 = "_grib2.bin"

#UTC時刻で入力
#today = datetime.date.today()
#data_year = (today - datetime.timedelta(days=1)).strftime('%Y')
#data_date = (today - datetime.timedelta(days=1)).strftime('%m%d')
data_year = 2023    #仮入力
data_date = "0129"    #仮入力

data_time = "120000"    #固定
time_diff = datetime.timedelta(hours=9) #時差


#ダウンロード用コード
#その時が来たら考える

#緯度指定
lat =36.106643
lon = 140.103164
#緯度 0.05度刻み
lat1 = lat - 0.025
lat2 = lat + 0.025
#経度 0.0625度刻み
lon1 = lon - 0.03125
lon2 = lon + 0.03125



#---------------------------------------------------------------------------------------------------------
#パラメータ定義
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
df = pd.DataFrame(columns=["year","month","day","hour","Pressure","temperature","u-component of wind","v-component of wind","Relative humidity","Total cloud cover"])

#関数：データ取得
def data_acquisition(data_year, data_date, data_time, data_range):

    ## GRIB2ファイルを読み込む
    dataname_base1 = "Z__C_RJTD_"
    dataname_base2 = "_MSM_GPV_Rjp_Lsurf_FH"
    dataname_base3 = "_grib2.bin"

    #ファイルオープン
    gpv_file = pygrib.open(dataname_base1 + str(data_year) + str(data_date) + data_time + dataname_base2 + data_range + dataname_base3)

    #ファイル抽出
    p_messages  = gpv_file.select(parameterName='Pressure')
    t_messages = gpv_file.select(parameterName='Temperature')
    uw_messages = gpv_file.select(parameterName='u-component of wind')
    vw_messages = gpv_file.select(parameterName='v-component of wind')
    rh_messages  = gpv_file.select(parameterName='Relative humidity')
    tcc_messages  = gpv_file.select(parameterName='Total cloud cover')
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
        ],
        "temperature": [
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] - 273.15 for msg in t_messages
        ],
        "u-component of wind":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in uw_messages
        ],
        "v-component of wind":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in vw_messages
        ],
        "Relative humidity":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in rh_messages
        ],
        "Total cloud cover":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in tcc_messages
        ]
    })

    df2 = pd.DataFrame({
        "radiation flux":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in dswrf_messages
        ]
    })

    #データ統合
    df_ = pd.concat([df_validdata, df1], axis=1)
    df_ = pd.concat([df_, df2], axis=1)
    df_.fillna(0)

    return df_

##16-33時間後(27-33時間後)データ
df_ = data_acquisition(data_year, data_date, data_time, data_range = "16-33")
df_.drop(range(0, 11),inplace=True)  #16-26時間後(対象日前日13:00～23:00)を削除
df_T = df_.T    #転置
list = (1, 3, 5, 7, 9, 11, 13)  #空の列挿入(毎時30分用)
for i in list:  
    df_T.insert(i, i + 0.5, np.nan)
df_ = df_T.T    #転置
df = pd.concat([df, df_], axis=0)


##34-39時間後データ
df_ = data_acquisition(data_year, data_date, data_time, data_range = "34-39")
df_T = df_.T    #転置
list = (1, 3, 5, 7, 9, 11)  #空の列挿入(毎時30分用)
for i in list:
    df_T.insert(i, i + 100.5, np.nan)
df_ = df_T.T    #転置
df = pd.concat([df, df_], axis=0)


##40-51時間後データ
df_ = data_acquisition(data_year, data_date, data_time, data_range = "40-51")

df_T = df_.T    #転置
list = (1, 3, 5, 7, 9, 11, 13, 15 ,17, 19, 21)  #空の列挿入(毎時30分用)
for i in list:
    df_T.insert(i, i + 200.5, np.nan)
df_ = df_T.T    #転置
df = pd.concat([df, df_], axis=0)


#index整理・線形補間
df = df.reset_index(drop=True)
df = df.interpolate()


##最終調整
#23.5時の値の処理
df.at[47, 'year'] = df.at[46, 'year']
df.at[47, 'month'] = df.at[46, 'month']
df.at[47, 'day'] = df.at[46, 'day']
df.at[47, 'hour'] = 23.5

#int化
df['year'] = df['year'].astype('int')
df['month'] = df['month'].astype('int')
df['day'] = df['day'].astype('int')

#51時間後(対象日翌日00:00)を削除
df = df.reset_index(drop=True)
df.drop(48,inplace=True)   

#出力
df.to_csv('weather_data.csv')
print(df)
