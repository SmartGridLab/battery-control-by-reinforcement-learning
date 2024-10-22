# 外部モジュール
import os.path
import warnings
import pandas as pd

# 内製モジュール
from RL_train import TrainModel
from RL_test import TestModel
from RL_dataframe_manager import Dataframe_Manager

## データベース
#import database_utils as db

class ChargeDischargePlan():
    def __init__(self, mode):
        warnings.simplefilter('ignore')
        print("\n---充放電計画策定プログラム開始---\n")
        self.trainModel = TrainModel() # 学習モデルのインスタンス化
        self.testModel = TestModel(mode) # テストモデルのインスタンス化
        self.dfmanager = Dataframe_Manager() # データベースのインスタンス化
        self.path = os.getcwd() + "/RL_trainedModels" 

    def mode_dependent_plan(self, mode):
        ### Training環境設定と実行
        # - もし、学習済みモデル(ex. 20240101.zip)がある場合は、Trainingをスキップして、Testを実行する
        # - 学習済みモデルがない場合は、Trainingを実行して、Testを実行する
        # もしpathにRL_trainedModelsというフォルダがない場合は、フォルダを作成する
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        # 学習済みモデル(zip)の全ファイル名をリストで取得
        model_list = os.listdir(self.path)


        if len(model_list) == 0: # /RL_trainedModelsにファイルがあるかどうかの確認
                    # 学習済みモデルがない場合は、TrainModelクラス内のdispatch_trainを実行する
                    print("-学習済みモデルがあるため、強化学習モデルのTrainingをスキップします-") 


        ### Test環境設定と実行&学習
        # model_listの中で最新のモデルを取得
        model_list.sort()
        latestModel_name = model_list[-1]
        # フォルダのpathを結合, lastModel_nameの.zipを削除して、.zipを除いたファイル名を取得 
        latestModel_name = self.path + "/" + latestModel_name.replace(".zip", "")
        # testを実行 (SoCとcharge/dischargeがリストに格納される)
        self.testModel.mode_dependent_test(latestModel_name, mode)


if __name__ == "__main__":
    main()
