# CSVファイルから学習データと予測子のデータを読み込む
# 前提：PV発電予測と価格予測の結果のcsvがあること

import pandas as pd

def getPriceImbPVout(table):
    # 30分単位のため、料金を0.5倍
    price = table["price"]/2   # [JPY/kWh] -> [JPY/kW/30min]
    imbalance = table["imbalance"]/2   # [JPY/kWh] -> [JPY/kW/30min]
    PVout = table["PVout"] # [kW]

    return price, imbalance, PVout

def RL_DataLoad():
    # データのロード
    print("-データロード-")
    # 学習データ
    input_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
    # テストデータ(これが充放電計画策定したいもの)
    predict_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv")

    # 学習データは、日数+1日分(48コマ分)のデータが必要なので、空データドッキングする
    data = [[0] * 20] * 48
    columns = ["year","month","day","hour","temperature","total precipitation","u-component of wind","v-component of wind","radiation flux","pressure","relative humidity","PVout","price","imbalance",
                "yearSin","yearCos","monthSin","monthCos","hourSin","hourCos"]
    new_rows_df = pd.DataFrame(data, columns=columns)
    input_data = input_data.append(new_rows_df, ignore_index=True)

    # 学習用データとテスト用データをテーブルに保存
    trainDataTable = getPriceImbPVout(input_data)
    predictorDataTable = getPriceImbPVout(predict_data)

    return trainDataTable, predictorDataTable
