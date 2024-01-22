#メインプログラム
import subprocess
import datetime

def main():
    print("\n---プログラム起動---\n")

    #時刻表示
    print("現在時刻：" + current_date.strftime("%Y/%m/%d") + " " + str(current_time) + "時")
    print("データ時刻:" + current_date.strftime("%Y/%m/%d") + " " + str(data_time) + "時")
    print("\nmode:" + mode +"\n")

    # # #天気予報データ取得(GPVデータ取得)
    # if mode == "bid":
    #     subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_bid.py', str(data_to_send)])
    # elif mode == "realtime":
    #     subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_realtime.py', str(data_to_send)])

    # #PV出力を予測する
    # subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])

    # #電力価格を予測する
    # subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])

    # Batteryの充放電計画を強化学習モデルで策定する
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_main.py'])
        # subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_main.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_main.py'])
        # subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_main.py'])

    # 充放電計画の性能評価のためのデータを集める
    # - PV発電量の実績値、電力価格の実績値、不平衡電力価格の実績値を取得する
    # - 実績値ベースでの売電による収益の計算を行う
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_inputdata_reference.py'])

    # 機器動作を策定する
    # - 強化学習で作られたcharge/discharge_realtime通りの充放電を実行しようとしてみる
    # - だけど、PVの予測値が外れたり、SoCの値がrealtime通りにならなかったりする
    # - bid_mode：入札したときの充放電計画(energytransfer_bid)に寄せて現実的な充放電を策定する-> charge/discharge_actual_bid
    # - realtime_mode：直前のコマの充放電計画(energytransfer_realtime)に寄せて現実的な充放電を策定する -> charge/discharge_actual_realtime
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate_bid.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate_realtime.py'])

    # 1日の最終コマ(23.5)の動作終了後に、1日分の結果を集計する
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_evaluration.py'])

    #終了
    print("\n---プログラム終了---\n")


# シミュレートする期間の選択
# OneDay：日付だけを指定して動作。1日分の強化学習モデルの動作結果を出す。
# OneTimeStep：単体時間(1コマ30分)を指定して動作（動作確認用という感じ）
simDuration = "OneDay"  #OneDay or OneTimeStep

# OneDayモード
if simDuration == "OneDay":
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
            # 0時の場合は bidモードとrealtime充放電計画策定も行う
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

                ## realtime modeを一時的に実行しないようにする (Jan 1st, 2024)-----------------------------------------------------------------------
                # # realtimeのときは、直前のコマ（前日の23.5）のデータを使って充放電計画を立てるので23.5時点のデータを使う
                # mode = "realtime"
                # print(mode)
                # data_time = 23.5
                # data_to_send = {'year': yesterday_date.year,'month': yesterday_date.month,'day': yesterday_date.day,'hour':data_time}
                # print(data_to_send)
                # main()
                ## --------------------------------------------------------------------------------------------------------------------------------
            
            ## realtime modeを一時的に実行しないようにする (Jan 1st, 2024)----------------------------------------------------------------------------
            # # 0時以外の場合は、realtimeモードのみで充放電計画策定を行う
            # else: 
            #     mode = "realtime"
            #     print(mode)
            #     data_time = current_time - 0.5
            #     data_to_send = {'year': current_date.year,'month': current_date.month,'day': current_date.day,'hour':data_time}
            #     print(data_to_send)
            #     main()
            ## --------------------------------------------------------------------------------------------------------------------------------
            # １コマ分時間を進める
            current_time += 0.5
        # １日分日付を進める
        current_date += datetime.timedelta(days=1)

# OneTimeStepモード
elif simDuration == "OneTimeStep":

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

    #時刻表示
    print("時刻：" + current_date.strftime("%Y/%m/%d") + " " + str(current_time) + "時")
    print("mode:" + mode)

    #天気予報データ取得
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_bid_test.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_realtime.py'])

    # PV出力予測：
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])

    # 電力価格予測：price_forecast.pyを実行する
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])

    # 強化学習による充放電スケジュール：RL_main.pyを実行する
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_main.py'])

    #終了
    print("\n---プログラム終了---\n")
