#メインプログラム
import subprocess
import datetime
import RL_operate
import pandas as pd

def perform_daily_operations(current_date, end_date):
    current_time = 0
    while current_date <= end_date:
        print("日付：" + current_date.strftime("%Y/%m/%d"))
        current_time = 0
        while current_time < 24:
            if current_time == 0:
                mode = "bid"
                print(mode)
                yesterday_date = current_date - datetime.timedelta(days=1)
                data_time = 0
                data_to_send = {'year': yesterday_date.year, 'month': yesterday_date.month, 'day': yesterday_date.day, 'hour': current_time}
                print(data_to_send)
                process_operations(mode, data_to_send)
            current_time += 0.5
        current_date += datetime.timedelta(days=1)

def process_operations(mode, data_to_send):
    # 複雑な操作をここに実装
    if mode == "bid":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_main.py'])
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_inputdata_reference.py'])
    elif mode == "realtime":
        subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/RL_main.py'])

    operate = RL_operate.Battery_operate()
    operate.operate_bid()
    subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_evaluration.py'])

# メインプログラム
def main():
    print("\n---プログラム起動---\n")
    simDuration = "MultipleDays"
    if simDuration == "MultipleDays":
        start_date = datetime.date(2022, 8, 8)
        end_date = datetime.date(2022, 8, 9)
        perform_daily_operations(start_date, end_date)
    print("\n---プログラム終了---\n")

if __name__ == "__main__":
    main()

