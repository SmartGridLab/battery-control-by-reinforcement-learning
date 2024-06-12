#メインプログラム
import subprocess
import datetime
from tracemalloc import start
import RL_operate
import pandas as pd

def perform_daily_operations(current_date, end_date):
    # 日付データを保持するためのリスト
    current_date_records = []
    current_time = 0
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
                yesterday_data_to_send = {'year': yesterday_date.year, 'month': yesterday_date.month, 'day': yesterday_date.day, 'hour': current_time}
                current_date_to_send = {'year': current_date.year, 'month': current_date.month, 'day': current_date.day, 'hour': current_time}
                print(current_date_to_send)


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
            


                current_date_records.append(current_date_to_send)
                # pandas DataFrame を作成
                df = pd.DataFrame(current_date_records)
                # current_date_to_send を CSV ファイルに保存
                df.to_csv('Battery-Control-By-Reinforcement-Learning/current_date.csv', index=False)
                print("Data saved to 'current_date.csv'")

                 # 操作を実行
                process_operations(mode)
            # １コマ分時間を進める
            current_time += 0.5
        current_date += datetime.timedelta(days=1)

        

def process_operations(mode):
    # 複雑な操作をここに実装

    # 充放電計画の性能評価のためのデータを集める
    # - PV発電量の実績値、電力価格の実績値、不平衡電力価格の実績値を取得する
    # - 実績値ベースでの売電による収益の計算を行う
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])
        print("price_predict success")
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])
        print("pv_predict success'")
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_main.py'])
        print("RL_main success'")
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_inputdata_reference.py'])
        print("result_inputdata_reference success'")
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_main.py'])

 # 機器動作を策定する
    # - 強化学習で作られたcharge/discharge_realtime通りの充放電を実行しようとしてみる
    # - だけど、PVの予測値が外れたり、SoCの値がrealtime通りにならなかったりする
    # - bid_mode：入札したときの充放電計画(energytransfer_bid)に寄せて現実的な充放電を策定する-> charge/discharge_actual_bid
    # - realtime_mode：直前のコマの充放電計画(energytransfer_realtime)に寄せて現実的な充放電を策定する -> charge/discharge_actual_realtime
    # Battery_operateをインスタンス化
    operate = RL_operate.Battery_operate()
     # RL_operate_bid.pyを実行する
    operate.operate_bid()
    # 1日の最終コマ(23.5)の動作終了後に、1日分の結果を集計する
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_evaluration.py'])

# メインプログラム
def main():
    print("\n---プログラム起動---\n")


    # # #天気予報データ取得(GPVデータ取得)
    # if mode == "bid":
    #     subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_bid.py', str(data_to_send)])
    # elif mode == "realtime":
    #     subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_realtime.py', str(data_to_send)])

    # #PV出力を予測する
    # subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])

    # #電力価格を予測する
    # subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])

    simDuration = "MultipleDays"
    if simDuration == "MultipleDays":
        # 動作開始日と動作終了日の指定
        # JST
        start_date = datetime.date(2022, 8, 1)
        end_date = datetime.date(2022, 8, 31)
        # 期間分の動作を実行
        perform_daily_operations(start_date, end_date)
        print("\n---プログラム終了---\n")

    # OneTimeStepモード
    if simDuration == "OneTimeStep":

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

if __name__ == "__main__":
    main()

