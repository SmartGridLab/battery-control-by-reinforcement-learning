#メインプログラム

import os
import subprocess
import datetime
import pytz
import time


#動作環境選択
#AIST：実際の時間に合わせて動作
#MULTI_TEST：複数時間を指定して動作
#SINGLE_TEST：単体時間を指定して動作
move_mode = "TEST"  #AIST or MULTI_TEST or SINGLE_TEST
# -> (小平)Multiとsingleは１つのモードに統合する（singleで動作させたければ、multiで開始と終了を同一時刻にする）

#モード選択
#bid：前日のスポット市場入札
#reaitime：当日のリアルタイム制御
# -> (小平)時刻で自動的にbidとrealtimeをケースわけして実行するように実装する

# タイムゾーンを設定
tz = pytz.timezone('Asia/Tokyo')

def main():
    print("\n---プログラム起動---\n")

    #時刻表示
    print("現在時刻：" + current_date.strftime("%Y/%m/%d") + " " + str(current_time) + "時")
    print("データ時刻:" + current_date.strftime("%Y/%m/%d") + " " + str(data_time) + "時")
    print("\nmode:" + mode +"\n")

    #天気予報データ取得
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_bid.py', str(data_to_send)])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_realtime.py', str(data_to_send)])

    #PV出力予測
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])

    #price_forecast.pyを実行する
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])

    # ESS_control.pyを実行する
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_schedule_dev.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_realtime_dev.py'])

    # dataframeの用意
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_dataframe_manager.py'])

    # 結果のdataframeへの書き込み
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_writing_bid.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_writing_plan.py'])

    # 過去データの参照
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_inputdata_reference.py'])

    # operate
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate_realtime.py'])

    # 評価
    if current_time == 23.5:
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_evaluration.py'])

    #終了
    print("\n---プログラム終了---\n")


#MULTI_TESTモード
if move_mode == "TEST":

    # 動作開始日と動作終了日の指定
    # JST
    # 現状複数日非対応
    start_date = datetime.date(2023, 1, 31)
    end_date = datetime.date(2023, 1, 31)

    #日付設定
    #current_date = start_date - datetime.timedelta(days=1)
    current_date = start_date
    current_time = 0
    # 動作
    data_to_send = {'year': current_date.year,'month': current_date.month,'day': current_date.day,'hour':current_time}


    while current_date <= end_date:
        print("日付：" + current_date.strftime("%Y/%m/%d"))
        
        #時間の初期値設定(hour表記)
        current_time = 0
        while current_time < 24:
            #print("現在時刻：" + str(current_time))
            #print("データ時刻:" + str(data_time))

            
            # 0時の場合は bidモードで充放電計画策定も行う
            if current_time == 0:
                mode = "bid"
                print(mode)

                yesterday_date = current_date - datetime.timedelta(days=1)
                data_time = 0
                data_to_send = {'year': yesterday_date.year,'month': yesterday_date.month,'day': yesterday_date.day,'hour':current_time}
                print(data_to_send)
                main()

                mode = "realtime"
                print(mode)
                data_time = 23.5
                data_to_send = {'year': yesterday_date.year,'month': yesterday_date.month,'day': yesterday_date.day,'hour':data_time}
                print(data_to_send)
                main()

            else: 
                mode = "realtime"
                print(mode)
                data_time = current_time - 0.5
                data_to_send = {'year': current_date.year,'month': current_date.month,'day': current_date.day,'hour':data_time}
                print(data_to_send)
                main()

            #print("データ時刻：" + str(data_time) + "時")

            #時間を変更
            current_time += 0.5
        current_date += datetime.timedelta(days=1)

#SINGLE_TESTモード
elif move_mode == "SINGLE_TEST":

    ## 手動で時刻を設定 ###
    year = 2023
    month = 1
    day = 1
    current_time = 23.5   #hour(0.5刻み) #bidの場合は0に設定
    mode = "realtime"   #reaitime　or bid
    #######################

    current_date = datetime.date(year, month, day)
    data_time = current_time - 0.5
    if data_time == -0.5:
        data_time = 23.5

    
    data_to_send = {'year': year,'month': month,'day': day,'hour':data_time}
    main()


