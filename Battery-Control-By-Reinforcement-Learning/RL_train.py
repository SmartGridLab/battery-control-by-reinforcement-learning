# 外部モジュール
import os
import warnings
from stable_baselines3 import PPO
import datetime

#from torch.utils.tensorboard import SummaryWriter # tensorBoardを起動して、学習状況を確認する

# 内製モジュール
from RL_env import ESS_ModelEnv as Env

warnings.simplefilter('ignore')
print("\n---充放電計画策定プログラム開始---\n")

class TrainModel:
    def __init__(self):
        # 現在のpathを取得して、モデルの保存先を指定
        self.path = os.getcwd() + "/RL_trainedModels"

    # モデルのトレーニング
    def dispatch_train(self):
        # 環境のインスタンス化
        self.env = Env()
        N_STEPS = 48 # 1日のコマ数
        N_EPISODES = 10 # 日数
        # n_steps: 1episode(1日)の中のコマ数。1コマ30分間なので、全部で48コマ。
        # total_timesteps: 学習全体での合計step数。n_steps * episode = total_timesteps
        print("-モデル学習開始-")
        model = PPO("MlpPolicy", self.env, gamma = 0.8, gae_lambda = 1, clip_range = 0.2, 
                    ent_coef = 0.005, vf_coef =0.5, learning_rate = 0.0001, n_steps = N_STEPS, 
                    verbose=0, tensorboard_log="./PPO_tensorboard/") 
        model.learn(total_timesteps= N_STEPS*N_EPISODES)  # これが大きすぎると、df_input等のデータが無い部分までenvの中でstate_idx参照してしまうのではないか？

        print("-モデル学習終了-")

        # 学習済みモデルの保存
        # - 保存ファイル名に現在日時(yyyy-mm-dd-hh-mm)を付与
        # - モデルの保存先は、/RL_trainedModels
        now = datetime.datetime.now()
        model_name = now.strftime("%Y-%m-%d-%H-%M")
        model.save(self.path + "/" + model_name)
                
        # 環境のクローズ
        self.env.close()

