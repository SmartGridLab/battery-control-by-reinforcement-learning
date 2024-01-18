# csvからのデータの読み込みを行うクラス

import pandas as pd

class Dataframe_Manager(): 
    ## 強化学習の学習に使うテーブル(df_train)を作成
    def get_train_df(self):
        # CSVファイル(input_data2022.csv)から学習データを読み込む
        # 読み込む行を列名で指定：year,month,day,hour, PVout, price, imbalance
        df_traindata = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv",
                                      usecols=["year","month","day","hour","PVout","price","imbalance"])        
        return df_traindata

    def get_test_df(self):
        # - 前提：PV発電予測と価格予測の結果のcsvがあること
        # - CSVファイルから予測結果のデータを読み込む
        # 電力価格データからyear, month, day,hour,price, imbalanceを読み込む
        price_predict = pd.read_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv", 
                                    usecols=["year","month","day","hour","price","imbalance"])
        # 列名を変更する
        price_predict = price_predict.rename(columns={'price': 'energyprice_predict', 'imbalance': 'imbalanceprice_predict'})
        # PV予測結果データからyear, month, day,hour, PVoutを読み込む
        pv_predict = pd.read_csv("Battery-Control-By-Reinforcement-Learning/pv_predict.csv",
                                    usecols=["year","month","day","hour","PVout"])
        # 列名を変更する
        pv_predict = pv_predict.rename(columns={'PVout': 'PV_predict'})
        # price_predictとpv_predictを結合（キーはyear,month,day,hourが全て一致） 
        df_testdata = pd.merge(price_predict, pv_predict, how='outer', on=['year','month','day','hour'])
        return df_testdata


    ## 強化学習の結果を入れるテーブル(df_result)を作成
    def get_resultform_df(self):
        # 列名をリストとして定義
        col = [
            # 時系列関係
            'year', 'month', 'day', 'hour',
            # PV関連
            'PV_predict_bid', 'PV_predict_realtime', 'PV_actual',
            # 電力価格関連
            'energyprice_predict_bid', 'energyprice_predict_realtime', 'energyprice_actual', 
            'imbalanceprice_predict_bid', 'imbalanceprice_predict_realtime', 'imbalanceprice_actual',
            # 充放電量関連
            'charge/discharge_bid', 'charge/discharge_realtime', 'charge/discharge_actual_bid', 'charge/discharge_actual_realtime',
            # SoC関連
            'SoC_bid', 'SoC_realtime', 'SoC_actual_bid', 'SoC_actual_realtime', 
            # 売電量関連
            'energytransfer_bid', 'energytransfer_realtime', 'energytransfer_actual_bid','energytransfer_actual_realtime',
            # 売電利益関連
            'energyprofit_bid', 'energyprofit_realtime', 'imbalancepenalty_bid', 'imbalancepenalty_realtime', 'totalprofit_bid', 'totalprofit_realtime',
            # 動作モード：operateの条件分岐を見るためのもの。デバッグ用
            'mode', 'mode_realtime'
        ]

        # 空のDataframeを作成
        dataframe = pd.DataFrame(columns=col)

        return dataframe

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


    # def get_data(self,idx):
    #     return  self.datafame(idx,:)


    def get_result_csv(self):
        # 表示
        print("dataframe作成")
        # DataframeをCSVファイルとして出力
        # 作成時間も同時に記入できるとよさそう？
        self.df_result.to_csv('./Battery-Control-By-Reinforcement-Learning/result_dataframe.csv', index=False)

    def write_