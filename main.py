import os
import subprocess

#スタート
print("\n\n---統合プログラム開始---\n\n")

#天気予報データ取得:OK
subprocess.run(['python', 'weather_data.py'])

#PV出力予測
subprocess.run(['python', 'pv_predict.py'])

#price_forecast.pyを実行する
subprocess.run(['python', 'price_predict.py'])

# ESS_control.pyを実行する
subprocess.run(['python', 'ESS_control.py'])

#終了
print("\n\n---統合プログラム終了---\n\n")