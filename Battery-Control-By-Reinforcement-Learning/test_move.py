#一括動作(テスト)
import os
import subprocess

#送るデータ
data_to_send = {
    'year': 2022,
    'month': 4,
    'day': 12,
    'hour':23.5
}

#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_bid.py', str(data_to_send)])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_realtime.py', str(data_to_send)])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_schedule_dev.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_realtime_dev.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_dataframe_manager.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_writing_bid.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_writing_plan.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_inputdata_reference.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate_realtime.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_evaluration.py'])