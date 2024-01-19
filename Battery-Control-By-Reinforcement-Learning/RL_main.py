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


warnings.simplefilter('ignore')
print("\n---充放電計画策定プログラム開始---\n")

trainModel = TrainModel() # 学習モデルのインスタンス化
testModel = TestModel() # テストモデルのインスタンス化
dfmanager = Dataframe_Manager() # データベースのインスタンス化

## Training環境設定と実行
# - もし、学習済みモデル(ex. 20240101.zip)がある場合は、Trainingをスキップして、Testを実行する
# - 学習済みモデルがない場合は、Trainingを実行して、Testを実行する
path = os.getcwd() + "/RL_trainedModels"   
# もしpathにRL_trainedModelsというフォルダがない場合は、フォルダを作成する
if not os.path.isdir(path):
    os.mkdir(path)
# 学習済みモデル(zip)の全ファイル名をリストで取得
model_list = os.listdir(path)

if len(model_list) == 0: # /RL_trainedModelsにファイルがあるかどうかの確認
    # 学習済みモデルがない場合は、TrainModelクラス内のdispatch_trainを実行する
    print("-学習済みモデルがないため、強化学習モデルのTrainingを実行します-")
    trainModel.dispatch_train() # trainを実行
    # 学習済みモデル(zip)の全ファイル名をリストで取得
    model_list = os.listdir(path)
else:
    # 学習済みモデルがある場合は、Trainingをスキップして、Testを実行する
    print("-学習済みモデルがあるため、強化学習モデルのTrainingをスキップします-") 

## Test環境設定と実行 学習
# model_listの中で最新のモデルを取得
model_list.sort()
latestModel_name = model_list[-1]
# フォルダのpathを結合, lastModel_nameの.zipを削除して、.zipを除いたファイル名を取得 
latestModel_name = path + "/" + latestModel_name.replace(".zip", "") 
# testを実行
obs_list, action_list = testModel.dispatch_test(latestModel_name) 
# obs_listとaction_listを書き込む
# list形式のobs_list, action_listをpandasのDataFrame形式に変換し、結合する
obs_list = pd.DataFrame(obs_list)
action_list = pd.DataFrame(action_list)
# 結合
df_test = pd.concat([obs_list, action_list], axis=1)
RL_dataframe_manager = dfmanager.write_result_csv(df_test)


