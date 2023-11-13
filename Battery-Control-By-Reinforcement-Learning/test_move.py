#一括動作用
import os
import subprocess

#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/weather_data_bid.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/pv_predict.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/price_predict.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_schedule_dev.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_dataframe_manager.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_writing_bid.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_writing_realtime.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_inputdata_reference.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate.py'])
#subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_evaluration.py'])