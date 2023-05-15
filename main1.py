import os
import subprocess

#スタート
print("\n\n---統合プログラム開始---\n\n")

#天気予報データ取得:OK
#subprocess.run(['python', 'weather_data.py'])

#PV出力予測
subprocess.run(['python', 'main.py'])

# price_forecast.pyを実行する
#subprocess.run(['python', 'price_forecast.py'])

# test.csvとpredicd_price.csvをマージしてESS_input.csvを生成する
#os.system('python -c "import pandas as pd; pd.concat([pd.read_csv(\'test.csv\'), pd.read_csv(\'predicd_price.csv\')], axis=1).to_csv(\'ESS_input.csv\', index=False)"')

# ESS_control.pyを実行する
#subprocess.run(['python', 'ESS_control.py'])

#終了
print("\n\n---統合プログラム終了---\n\n")