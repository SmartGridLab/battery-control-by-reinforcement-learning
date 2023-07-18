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
move_mode = "SINGLE_TEST"  #AIST or MULTI_TEST or SINGLE_TEST
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
    print("時刻：" + current_date.strftime("%Y/%m/%d") + " " + str(current_time) + "時")
    print("mode:" + mode)

    #天気予報データ取得
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_bid_test.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_realtime.py'])

    #PV出力予測
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])

    #price_forecast.pyを実行する
    #subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])

    # ESS_control.pyを実行する
    #subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_control.py'])

    #終了
    print("\n---プログラム終了---\n")


#AISTモード
if move_mode == "AIST":

    #無限ループ
    while True:
    # プログラムの実行コードをここに書く
        now = datetime.datetime.now(tz)
        current_date = now.date()
        current_time = now.hour
        current_time = int(current_time)

        #毎時30分を超えた場合、current_timeをn.5時になるよう設定
        current_minute = now.minute
        current_minute = int(current_minute)
        if current_minute >= 30:
            current_time += 0.5

        #時間ごとにモードを指定し、main.pyを動作
        #mode = "bid"
        subprocess.run(['python', 'main.py'])

        # 30分待機する
        time.sleep(1800)

#MULTI_TESTモード
elif move_mode == "TEST":

    # 動作開始日と動作終了日の指定
    # JST
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 3)

    # 日付の範囲で動作

    #日付設定
    current_date = start_date - datetime.timedelta(days=1)
    current_time = 0
    mode = "bid"
    print(mode)
    #subprocess.run(['python', 'main.py'])


    while current_date <= end_date:
        print("日付：" + current_date.strftime("%Y/%m/%d"))
        
        #時間の初期値設定(hour表記)
        current_time = 0
        while current_time < 24:
            print("時刻：" + str(current_time))
            
            # 0時の場合は bidモードで充放電計画策定も行う
            if current_time == 0:
                mode = "bid"
                print(mode)
                #subprocess.run(['python', 'main.py'])

            mode = "realtime"
            print(mode)
            #subprocess.run(['python', 'main.py'])

            #時間を変更
            current_time += 0.5
        current_date += datetime.timedelta(days=1)

#SINGLE_TESTモード
elif move_mode == "SINGLE_TEST":

    ## 手動で時刻を設定
    current_date = datetime.date(2023, 7, 6) #(YYYY, MM, DD)
    current_time = 16   #hour(0.5刻み)
    mode = "realtime"   #reaitime　or bid

    main()


