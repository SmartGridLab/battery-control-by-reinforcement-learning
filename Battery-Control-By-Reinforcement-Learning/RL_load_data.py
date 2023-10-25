# CSVファイルから学習データと予測子のデータを読み込む
# 前提：PV発電予測と価格予測の結果のcsvがあること

import pandas as pd

class LoadData:
    def __init__(self):
        # 学習データ
        self.trainData = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
        # テストデータ(これが充放電計画策定したいもの)
        self.predictorData = pd.read_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv")
        self.trainDataTable
        self.predictorDataTable

    # 学習データと予測子のデータをテーブル形式で返す
    def get_DataTable(self):
        # 学習データは、日数+1日分(48コマ分)のデータが必要なので、空データドッキングする
        data = [[0] * 20] * 48
        columns = ["year","month","day","hour","temperature","total precipitation","u-component of wind","v-component of wind","radiation flux","pressure","relative humidity","PVout","price","imbalance",
                    "yearSin","yearCos","monthSin","monthCos","hourSin","hourCos"]
        new_rows_df = pd.DataFrame(data, columns=columns)
        input_data = self.trainData.append(new_rows_df, ignore_index=True)

        # 学習用データとテスト用データをテーブルに保存
        self.trainDataTable = getPriceImbPVout(input_data)
        self.predictorDataTable = getPriceImbPVout(predict_data)

        return self.trainDataTable, self.predictorDataTable

    # 
    def get_PriceImbPVout(table):
        # 30分単位のため、料金を0.5倍
        price = table["price"]/2   # [JPY/kWh] -> [JPY/kW/30min]
        imbalance = table["imbalance"]/2   # [JPY/kWh] -> [JPY/kW/30min]
        PVout = table["PVout"] # [kW]

        return price, imbalance, PVout
