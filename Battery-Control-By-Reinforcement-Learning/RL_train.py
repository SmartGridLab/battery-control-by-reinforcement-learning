# 外部モジュール
import os
import warnings
import datetime
import pytz
import matplotlib.pyplot as plt  # 追加
import datetime
from stable_baselines3 import PPO
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
#from torch.utils.tensorboard import SummaryWriter # tensorBoardを起動して、学習状況を確認する

# 内製モジュール
from RL_env import ESS_ModelEnv as Env_bid
from RL_env_realtime import ESS_ModelEnv as Env_realtime
import pandas as pd

warnings.simplefilter('ignore')
print("\n---充放電計画策定プログラム開始---\n")

class TQDMProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        if self.pbar is not None:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()

class TrainModel:
    def __init__(self):
        # 現在のpathを取得して、モデルの保存先を指定
        self.path = os.getcwd() + "/RL_trainedModels"

    def plot_episode_info(self, mode):
        # subplot設定
        subplot_a = 4

        ## グラフのスタイル設定
        # color
        color_rewards = "blue"
        color_imbalance = "orange"
        color_DealProfit  = "red"
        color_ActionDifference = "green"
        # linestyle
        linestyle_rewards = "-"
        linestyle_imbalance = "-"
        linestyle_DealProfit = "-"
        linestyle_ActionDifference = "-"
        # label
        label_rewards = "Episode rewards"
        label_imbalance = "Imbalance Cost"
        label_DealProfit = "Deal Profit"
        label_ActionDifference = "Action_Difference"

        plt.figure(figsize = (60, 50))
        # 1. RLの報酬のグラフ
        plt.subplot(subplot_a, 1, 1)
        plt.plot(self.env.episode_rewards_summary, color = color_rewards, linestyle = linestyle_rewards, label = label_rewards)
        plt.title("Episode rewards")
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend(loc = "upper right")
        plt.grid(True)
        # 2. インバランスコストのグラフ
        plt.subplot(subplot_a, 1, 2)
        plt.plot(self.env.imbalance_summary, color = color_imbalance, linestyle = linestyle_imbalance, label = label_imbalance)
        plt.title("Imbalance Cost")
        plt.xlabel("Episode")
        plt.ylabel("Imbalance Cost [Yen]")
        plt.legend(loc = "upper right")
        plt.grid(True)
        # 3. 取引収益のグラフ
        plt.subplot(subplot_a, 1, 3)
        plt.plot(self.env.episode_action_summary, color = color_DealProfit, linestyle = linestyle_DealProfit, label = label_DealProfit)
        plt.title("Episode Action Summary")
        plt.xlabel("Episode")
        plt.ylabel("Action Summary [kWh]")
        plt.legend(loc = "upper right")
        plt.grid(True)
        # 4. Action Differenceのグラフ
        plt.subplot(subplot_a, 1, 4)
        plt.plot(self.env.action_difference_summary, color = color_ActionDifference, linestyle = linestyle_ActionDifference, label = label_ActionDifference)
        plt.title("Action difference")
        plt.xlabel("Episode")
        plt.ylabel("Action difference [kWh]")
        plt.legend(loc = "upper right")
        plt.grid(True)

        # グラフの表示と保存
        plt.tight_layout()  # レイアウトの自動調整
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"Battery-Control-By-Reinforcement-Learning/for_debug/RLTrain/{mode}/RL_info_{current_time}.png")
        plt.show()

    def plot_action_difference(self, mode):
        plt.figure()
        plt.plot(self.env.action_difference_summary)
        plt.title("Action Difference with the reward $r_1$", fontsize = 16)
        plt.xlabel("Episode", fontsize = 14)
        plt.ylabel("Action Difference [kWh]", fontsize = 14)
        plt.grid(True)
        # グラフの表示と保存
        plt.tight_layout()  # レイアウトの自動調整
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"Battery-Control-By-Reinforcement-Learning/for_debug/RLTrain/{mode}/RL_info_{current_time}_action_difference.png")
        plt.show()

    def dispatch_train(self, mode):
        # 環境のインスタンス化
        if mode == "bid":
            self.env = Env_bid(mode)
        elif mode == "realtime":
            self.env = Env_realtime(mode)

        N_STEPS = 48  # 1日のコマ数（=1エピソード内のステップ数）
        N_EPISODES = 30000  # 学習する日数（エピソード数
        total_timesteps = N_STEPS * N_EPISODES  # エピソード数に基づいた総ステップ数
        # n_steps: 1episode(1日)の中のコマ数。1コマ30分間なので、全部で48コマ。
        # total_timesteps: 学習全体での合計step数。n_steps * episode = total_timesteps
        print("-モデル学習開始-")
        model = PPO("MlpPolicy", self.env, gamma = 0.8, gae_lambda = 1, clip_range = 0.2, 
                    ent_coef = 0.005, vf_coef =0.5, learning_rate = 0.0001, n_steps = N_STEPS, 
                    verbose=0, tensorboard_log="./PPO_tensorboard/") 
        progress_bar_callback = TQDMProgressBarCallback(total_timesteps = total_timesteps)
        model.learn(total_timesteps = total_timesteps, callback = progress_bar_callback)

        print("-モデル学習終了-")
        print(len(self.env.episode_rewards_summary))

        # 学習済みモデルの保存
        # - 保存ファイル名にJSTでの現在日時(yyyy-mm-dd-hh-mm)を付与
        # - モデルの保存先は、/RL_trainedModels
        # - モデルの保存形式は、.zip
        now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        model_name = now.strftime("%Y-%m-%d-%H-%M")
        model.save(self.path + f"/{mode}/" + model_name)    
                
        # 環境のクローズ
        self.env.close()

        # エピソードごとのリワードをグラフで表示(追加)
        self.plot_episode_info(mode)
        self.plot_action_difference(mode)


# メイン関数
if __name__ == "__main__":
    mode = "bid"
    # mode = "realtime"
    trainer = TrainModel()
    trainer.dispatch_train(mode)

    print("\n---充放電計画策定プログラム終了---\n")