#メインプログラム

import os
import subprocess
import datetime
import pytz
import time


# 動作環境選択
# TEST：日付だけを指定して動作 ->　79行目以降指定
# SINGLE_TEST：単体時間(1コマ30分)を指定して動作（動作確認用という感じ）
# -> (小平)Multiとsingleは１つのモードに統合する（singleで動作させたければ、multiで開始と終了を同一時刻にする）
move_mode = "TEST"  #TEST or SINGLE_TEST

# タイムゾーンを設定
tz = pytz.timezone('Asia/Tokyo')

def main():
    print("\n---プログラム起動---\n")

    #時刻表示
    print("現在時刻：" + current_date.strftime("%Y/%m/%d") + " " + str(current_time) + "時")
    print("データ時刻:" + current_date.strftime("%Y/%m/%d") + " " + str(data_time) + "時")
    print("\nmode:" + mode +"\n")

    #天気予報データ取得(GPVデータ取得)
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_bid.py', str(data_to_send)])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_realtime.py', str(data_to_send)])

    #PV出力を予測する
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])

    #電力価格を予測する
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])

    # Batteryの充放電計画を強化学習モデルで策定する
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_schedule.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_realtime.py'])

    # dataframeの用意
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_dataframe_manager.py'])

    # 結果のdataframeへの書き込み
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_writing_bid.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_writing_realtime.py'])

    # 充放電計画の性能評価のためのデータを集める
    # - PV発電量の実績値、電力価格の実績値、不平衡電力価格の実績値を取得する
    # - 実績値ベースでの売電による収益の計算を行う
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_inputdata_reference.py'])

    # 機器動作を策定する
    # - 強化学習で作られたplanを実行しようとしてみる
    # - だけど、PVの予測値が外れたり、SoCの値がplan通りにならなかったりする
    # - bid_mode：入札したときの充放電計画(energytransfer_bid)に寄せて現実的な充放電を策定する
    # - realtime_mode：直前のコマの充放電計画(energytransfer_plan)に寄せて現実的な充放電を策定する
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate_bid.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate_realtime.py'])

    # 評価
    if current_time == 23.5:
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_evaluration.py'])

    #終了
    print("\n---プログラム終了---\n")


# TESTモード
if move_mode == "TEST":
    # 動作開始日と動作終了日の指定
    # JST
    # 現状複数日非対応
    # ex) 2022年8月8日のみ動作:(2022, 8, 8)
    start_date = datetime.date(2022, 8, 8)
    end_date = datetime.date(2022, 8, 8)

    # 日付と時刻のカウンタ設定
    # current_date = start_date - datetime.timedelta(days=1)
    current_date = start_date   # 日付の初期値設定
    current_time = 0            # 時刻の初期値設定(0.5刻み,  max23.5)

    while current_date <= end_date:
        print("日付：" + current_date.strftime("%Y/%m/%d"))
        
        #時間の初期値設定(hour表記)
        current_time = 0
        while current_time < 24:
            # 0時の場合は bidモードで充放電計画策定も行う
            if current_time == 0:
                mode = "bid"
                print(mode)
                # bidのときは前日の10AM時点で手に入るデータで入札をしていることになるので、そのスケジュールを生成するために実行する
                yesterday_date = current_date - datetime.timedelta(days=1)
                data_time = 0
                # GPVデータの所得のために時刻を生成
                data_to_send = {'year': yesterday_date.year,'month': yesterday_date.month,'day': yesterday_date.day,'hour':current_time}
                print(data_to_send)
                main()
                # realtimeのときは、直前のコマ（前日の23.5）のデータを使って充放電計画を立てる
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

            # １コマ分時間を進める
            current_time += 0.5
        # １日分日付を進める
        current_date += datetime.timedelta(days=1)

# SINGLE_TESTモード
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


