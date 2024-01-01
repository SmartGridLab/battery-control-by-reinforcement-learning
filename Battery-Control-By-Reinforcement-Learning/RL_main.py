# 外部モジュール
import os.path
import warnings

# 内製モジュール
from RL_env import ESS_ModelEnv
from RL_train import TrainModel
from RL_test import TestModel

warnings.simplefilter('ignore')
print("\n---充放電計画策定プログラム開始---\n")

# 実行部分
if __name__ == "__main__" :
    ## パラメーター設定
    pdf_day = 0 #確率密度関数作成用のDay数 75 80
    train_days = 366 # 学習Day数 70 ~ 73
    test_day = 3 # テストDay数 + 2 (最大89)
    PV_parameter = "PVout" # Forecast or PVout_true (学習に使用するPV出力値の種類)　#今後はUpper, lower, PVout
    episode = 6 # 10000000    # 学習回数

    ## Training環境設定と実行
    # - もし、学習済みモデル(ex. 20240101.zip)がある場合は、Trainingをスキップして、Testを実行する
    # - 学習済みモデルがない場合は、Trainingを実行して、Testを実行する
    path = os.getcwd() + "/RL_trainedModels"   
    model_list = os.listdir(path)
    if len(model_list) == 0: # /RL_trainedModelsにファイルがあるかどうかの確認
        # 学習済みモデルがない場合は、Trainingを実行して、Testを実行する
        env = ESS_ModelEnv(train_days, test_day)
        TrainModel.dispatch_train(env) # trainを実行
    else:
        # 学習済みモデルがある場合は、Trainingをスキップして、Testを実行する
        print("-学習済みモデルがあるため、Trainingをスキップします-") 

    ## Test環境設定と実行 学習
    # model_listの中で最新のモデルを取得
    model_list.sort()
    latestModel_name = model_list[-1]
    env = ESS_ModelEnv(train_days, test_day)
    TestModel.dispatch_test(latestModel_name) # testを実行

    
