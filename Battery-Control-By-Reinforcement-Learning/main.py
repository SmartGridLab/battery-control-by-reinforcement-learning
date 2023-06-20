import os
import subprocess

#スタート
print("\n\n---統合プログラム開始---\n\n")

#天気予報データ取得:OK
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data.py'])

#PV出力予測
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])

#price_forecast.pyを実行する
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])

# ESS_control.pyを実行する
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_control_test_0129.py'])

#終了
print("\n\n---統合プログラム終了---\n\n")