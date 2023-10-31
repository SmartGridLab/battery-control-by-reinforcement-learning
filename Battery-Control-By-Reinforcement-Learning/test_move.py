#一括動作用
import os
import subprocess

subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_dataframe_manager.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_writing_bid.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_inputdata_reference.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/ESS_operate.py'])
subprocess.run(['python', 'Battery-Control-By-Reinforcement-Learning/result_evaluration.py'])