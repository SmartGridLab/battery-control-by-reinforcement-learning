#メインプログラム
import subprocess
import datetime
from tracemalloc import start
from RL_operate import Battery_operate
from RL_main import ChargeDischargePlan
from result_evaluation import ResultEvaluation
from pv_predict import PV_Predict
import RL_visualize
import pandas as pd

def perform_daily_operations(current_date, end_date, mode):
    while current_date <= end_date:
        print(mode)
        print("current_date: ", current_date)
        # current_date.csvに保存するための時刻データを作成
        # current_date を年、月、日別々の列として保存
        current_date_df = pd.DataFrame([[current_date.year, current_date.month, current_date.day, current_date.hour]], columns=['year', 'month', 'day', 'hour'])
        # current_date_csv を CSV ファイルに保存（ここで取得した日付を現在日時としてたファイルでも使う）
        current_date_df.to_csv('Battery-Control-By-Reinforcement-Learning/current_date.csv', index=False)

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

        # 操作を実行
        process_operations(mode)
        current_date += datetime.timedelta(days=1)

def process_operations(mode):
    # 種々のプログラム実行

    # 充放電計画の性能評価のためのデータを集める
    # - PV発電量の実績値、電力価格の実績値、不平衡電力価格の実績値を取得する
    # - 実績値ベースでの売電による収益の計算を行う
    #　それぞれのファイルでcurrent_date.csvの日付データに基づいてデータを取得
    #　気象データを取得
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_bid.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_realtime.py'])
        print("weather_data_bid success")

    # 電力価格を予測
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])
    print("price_predict success")
    # PV出力を予測
    PV_Predict().mode_dependent_pv_predict(mode)
    print("pv_predict success'")
    # 強化学習による充放電スケジュールを作成
    ChargeDischargePlan(mode).mode_dependent_plan(mode)
    print("RL_main success'")
    # 充放電計画の性能評価のためのデータを集める
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_inputdata_reference.py'])
    print("result_inputdata_reference success'")
    Battery_operate().mode_dependent_operate(mode) # RL_operateのインスタンス化
    print("RL_operate success'")
    # 充放電計画の性能評価を行う
    ResultEvaluation().evaluation_result_save(mode)
    print("result_evaluation success'")
    # 収益を棒グラフで可視化して比較
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_visualize_bargraph.py'])
    print("RL_visualize_bargraph success'")

    # 機器動作を策定する
    # - 強化学習で作られたcharge/discharge_realtime通りの充放電を実行しようとしてみる
    # - だけど、PVの予測値が外れたり、SoCの値がrealtime通りにならなかったりする
    # - bid_mode：入札したときの充放電計画(energytransfer_bid)に寄せて現実的な充放電を策定する-> charge/discharge_actual_bid
    # - realtime_mode：直前のコマの充放電計画(energytransfer_realtime)に寄せて現実的な充放電を策定する -> charge/discharge_actual_realtime
  
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

    ##-------------------------------------------------------------------------------------------------------------------------##
    simDuration = "MultipleDays_FullMode" # "MultipleDays_SingleMode" = 単一実行モード、"MultipleDays_FullMode" = 連続実行モード
    ##-------------------------------------------------------------------------------------------------------------------------##

    if simDuration == "MultipleDays_SingleMode":
        print("\n---Single Mode プログラム開始---\n")
        # 動作開始日と動作終了日の指定
        # JST
        start_date = datetime.datetime(2022, 9, 1, 0, 30)
        end_date = datetime.datetime(2022, 9, 2, 23, 30)
        # 期間分の動作を実行
        # modeを指定して実行
        # mode = "bid"
        mode = "realtime"
        perform_daily_operations(start_date, end_date, mode)
        print(f"\n---{mode} プログラム終了---\n")
        print("\n---Single Mode プログラム終了---\n")

    if simDuration == "MultipleDays_FullMode":
        print("\n---Full Mode プログラム開始---\n")
        # 動作開始日と動作終了日の指定
        # JST
        start_date = datetime.datetime(2022, 9, 1, 0, 30)
        end_date = datetime.datetime(2022, 9, 2, 23, 30)
        # 期間分の動作を実行
        perform_daily_operations(start_date, end_date, "bid")
        print("\n---Bid プログラム終了---\n")
        perform_daily_operations(start_date, end_date, "realtime")
        print("\n---Realtime プログラム終了---\n")
        print("\n---Full Mode プログラム終了---\n")

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
    fig = RL_visualize.RL_visualize.descr_price()