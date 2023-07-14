import urllib.request
import os

import datetime
import pytz
import pygrib
import pandas as pd
import numpy as np


print("---気象予報データ抽出プログラム開始---\n")

# 日時取得########################################################
# 開始日と終了日の指定
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2023, 1, 31)


# Grid Point指定#########################
#欲しい場所の場所指定
lat =36.06489716079195
lon = 140.1349848817127

#最寄りのGrid Point探索用の範囲指定
#緯度の最小格子：0.05度刻み
lat1 = lat - 0.025
lat2 = lat + 0.025
#経度の最小格子：0.0625度刻み
lon1 = lon - 0.03125
lon2 = lon + 0.03125

#最初に表示(確認用)
print("緯度 : " + str(lat))
print("経度 : " + str(lon) + "\n")

#時差
time_diff = datetime.timedelta(hours=9) #pygribに使用



#出力データの型枠生成
df = pd.DataFrame(columns=["year","month","day","hour","Pressure","temperature","u-component of wind","v-component of wind","Relative humidity", "Total cloud cover", "Total precipitation"])


#ファイル名設定###################################################################################


# Grid Point指定#########################
#欲しい場所の場所指定
lat =36.06489716079195
lon = 140.1349848817127

#最寄りのGrid Point探索用の範囲指定
#緯度の最小格子：0.05度刻み
lat1 = lat - 0.025
lat2 = lat + 0.025
#経度の最小格子：0.0625度刻み
lon1 = lon - 0.03125
lon2 = lon + 0.03125


# 表示部分################################################################
#最初に表示(確認用)
print("緯度 : " + str(lat))
print("経度 : " + str(lon) + "\n")



#---------------------------------------------------------------------------------------------------------
#GPVデータパラメータ定義
#参考サイト
#https://predora005.hatenablog.com/entry/2020/10/31/000000
#https://qiita.com/kurukuruz/items/6fc0be9efa34a2fd6741

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



##関数：データ取得
def data_acquisition(data_year, data_date, data_time, data_date1, data_range):

    # GPVファイル(.grib2)を読み込む#########################################

    #GPVファイルのファイル名指定
    #このプログラムのフォルダ名
    dataname_base = "Battery-Control-By-Reinforcement-Learning/"

    #GPVファイル名
    dataname_base1 = "Z__C_RJTD_"
    dataname_base2 = "_MSM_GPV_Rjp_Lsurf_FH"
    dataname_base3 = "_grib2.bin"

    #GPVファイル名(ダウンロード用)
    file_name = dataname_base1 + str(data_year) + str(data_date) + data_time + dataname_base2 + data_range + dataname_base3

    #ファイルパス(フォルダ指定用)
    file_path = dataname_base + file_name


    ## 京大RISHからファイルをダウンロード
    print(data_range +"時間後予測  ファイルダウンロード開始...")
    url_surf = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/" + str(data_date1) + "/" + file_name
    urllib.request.urlretrieve(url_surf, file_path)
    print(data_range +"時間後予測  ファイルダウンロード完了")


    #ファイルオープン
    gpv_file = pygrib.open(file_path)
    print(data_range +"時間後予測  データ取得開始...")
    

    # データ抽出#########################################################3
    #パラメータ指定
    p_messages  = gpv_file.select(parameterName='Pressure')
    t_messages = gpv_file.select(parameterName='Temperature')
    uw_messages = gpv_file.select(parameterName='u-component of wind')
    vw_messages = gpv_file.select(parameterName='v-component of wind')
    rh_messages  = gpv_file.select(parameterName='Relative humidity')
    tcc_messages  = gpv_file.select(parameterName='Total cloud cover')
    tp_messages  = gpv_file.select(parameterName='Total precipitation')
    dswrf_messages  = gpv_file.select(parameterName='Downward short-wave radiation flux')

    #時系列取り出し・時系列データ分解
    df_validdata_ = pd.DataFrame({"validDate": [msg.validDate + time_diff for msg in t_messages]})
    df_validdata = pd.DataFrame(columns=["year","month","day","hour"])
    df_validdata['year'] = df_validdata_['validDate'].dt.year
    df_validdata['month'] = df_validdata_['validDate'].dt.month
    df_validdata['day'] = df_validdata_['validDate'].dt.day
    df_validdata['hour'] = df_validdata_['validDate'].dt.hour 

    #各データフレームへデータ格納
    #気圧([hPa]へ変換)
    df1 = pd.DataFrame({
        "Pressure":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] * 0.01 for msg in p_messages
        ]
   })
    #気温(摂氏変換)
    df2 = pd.DataFrame({
        "temperature": [
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] - 273.15 for msg in t_messages
        ]
    })
    #u風速(東西方向)
    df3 = pd.DataFrame({
        "u-component of wind":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in uw_messages
        ]
    })
    #v風速(南北方向)
    df4 = pd.DataFrame({
        "v-component of wind":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in vw_messages
        ]
    })
    #湿度
    df5 = pd.DataFrame({
        "Relative humidity":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in rh_messages
        ]
    })
    #雲量
    df6 = pd.DataFrame({
        "Total cloud cover":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in tcc_messages
        ]
    })
    #降水量
    df7 = pd.DataFrame({
        "Total precipitation":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in tp_messages
        ]
    })
    #日射量
    df8 = pd.DataFrame({
        "radiation flux":[
            msg.data(lat1, lat2, lon1, lon2)[0][0][0] for msg in dswrf_messages
        ]
    })

    # データ整理####################################################
    #データフレーム統合
    df_ = pd.concat([df_validdata, df1], axis=1)
    df_ = pd.concat([df_, df2], axis=1)
    df_ = pd.concat([df_, df3], axis=1)
    df_ = pd.concat([df_, df4], axis=1)
    df_ = pd.concat([df_, df5], axis=1)
    df_ = pd.concat([df_, df6], axis=1)
    df_ = pd.concat([df_, df7], axis=1)
    df_ = pd.concat([df_, df8], axis=1)
    
    #欠損値へ0を挿入
    df_.fillna(0)

    print(data_range +"時間後予測  データ取得完了")

    #ファイルクローズ
    gpv_file.close()
    #ファイル削除
    os.remove(file_path)

    print(data_range +"時間後予測  ファイル削除完了\n")

    return df_

##関数：1日分の処理
def cycle_oneday(current_date):
    df_ = pd.DataFrame(columns=["year","month","day","hour","Pressure","temperature","u-component of wind","v-component of wind","Relative humidity", "Total cloud cover", "Total precipitation"])

    #ファイル名設定###################################################################################
    #DLするGPVファイルの日付設定
    data_year = (current_date - datetime.timedelta(days=2)).strftime("%Y")
    data_date = (current_date - datetime.timedelta(days=2)).strftime("%m%d")
    #公開先URLの設定
    data_date1 = (current_date - datetime.timedelta(days=2)).strftime("%Y/%m/%d")
    #DLするGPVファイルの時刻(予測)設定
    data_time = "120000"    #スポット市場用は常に固定(21時(JST)公開の78時間予測を使用するため)

    print(str(data_date1) + " 21:00(JST)/12:00(UTC)公開のデータを取得\n")

    # 公開されているファイルごとに処理を実行##################################################################
    # 16-33時間後(27-33時間後)予測のファイルを処理#############################################
    df__ = data_acquisition(data_year, data_date, data_time, data_date1, data_range = "16-33")   #データ抽出

    df__.drop(range(0, 11),inplace=True)  #出力範囲に含まない16-26時間後(対象日前日13:00～23:00)を削除
    df__T = df__.T    #空の列を挿入するために転置(毎時30分用)
    list = (1, 3, 5, 7, 9, 11, 13)  #空の列挿入(毎時30分用)
    for i in list:  
        df__T.insert(i, i + 0.5, np.nan)    #index番号の重複を避けるためi + 0.5 とする
    df__ = df__T.T    #転置して元に戻す
    df_ = pd.concat([df_, df__], axis=0)   #出力用データフレームに統合


    # 34-39時間後データ予測のファイルを処理####################################################
    df__ = data_acquisition(data_year, data_date, data_time, data_date1, data_range = "34-39")   #データ抽出

    df__T = df__.T    #空の列を挿入するために転置(毎時30分用)
    list = (1, 3, 5, 7, 9, 11)  #空の列挿入(毎時30分用)
    for i in list:
        df__T.insert(i, i + 0.5, np.nan)    #index番号の重複を避けるためi + 0.5 とする
    df__ = df__T.T    #転置して元に戻す
    df_ = pd.concat([df_, df__], axis=0)   #出力用データフレームに統合


    # 40-51時間後データ予測のファイルを処理####################################################
    df__ = data_acquisition(data_year, data_date, data_time, data_date1, data_range = "40-51")   #データ抽出

    df__T = df__.T    #空の列を挿入するために転置(毎時30分用)
    list = (1, 3, 5, 7, 9, 11, 13, 15 ,17, 19, 21)  #空の列挿入(毎時30分用)
    for i in list:
        df__T.insert(i, i + 0.5, np.nan)   #index番号の重複を避けるためi + 0.5 とする
    df__ = df__T.T    #転置して元に戻す
    df_ = pd.concat([df_, df__], axis=0)   #出力用データフレームに統合


    # データ整理######################################################
    #index整理(1から振りなおす)
    df_ = df_.reset_index(drop=True)

    #線形補間
    df_ = df_.interpolate()

    # 線形補間後の数値調整
    #23.5時の値の処理
    df_.at[47, 'year'] = df_.at[46, 'year']
    df_.at[47, 'month'] = df_.at[46, 'month']
    df_.at[47, 'day'] = df_.at[46, 'day']
    df_.at[47, 'hour'] = 23.5

    #転置の時に年月日がdouble型になっているため、int化
    df_['year'] = df_['year'].astype('int')
    df_['month'] = df_['month'].astype('int')
    df_['day'] = df_['day'].astype('int')

    #48個のデータにするために51時間後(対象日翌日00:00)を削除
    df_ = df_.reset_index(drop=True)
    df_.drop(48,inplace=True)

    return df_

current_date = start_date
while current_date <= end_date:
    print(current_date.strftime("%Y/%m/%d"))
    df_ =cycle_oneday(current_date)
    df = pd.concat([df, df_], axis=0)
    #print(df)
    current_date += datetime.timedelta(days=1)

# ファイル出力###############################################################
df.to_csv('Battery-Control-By-Reinforcement-Learning/weather_data_bid.csv')
print("--結果出力完了--")
#print(df)
print("\n\n---気象予報データ抽出プログラム終了---")