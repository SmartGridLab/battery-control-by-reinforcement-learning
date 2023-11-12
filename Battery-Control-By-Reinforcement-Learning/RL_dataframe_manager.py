import pandas as pd

Class Dataframe_Manager():
    def __init__(self):
        ## 強化学習の学習に使うテーブル(df_input)を作成
        # - 前提：PV発電予測と価格予測の結果のcsvがあること
        # - CSVファイルから学習データと予測子のデータを読み込む

        # 電力価格データ（過去＋予測）
        price_predict = pd.read_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv")
        # PV, wind予測結果データ（過去＋予測）
        pv_wind_predict = pd.read_csv("Battery-Control-By-Reinforcement-Learning/pv_wind_predict.csv")
        # 電力価格とPV/windテーブルを結合（キーはyear, month, day, hour　が全て一致）
        self.df_input = pd.merge(price_predict, pv_wind_predict, how='outer', on=['year','month','day','hour'])

        ## 強化学習の結果を入れるテーブル(df_result)を作成
        # 列名をリストとして定義
        # wind_q10: Quantile Regressionによる10%分位点の風力発電の発電量[MWh]の予測結果
        col = [
            'year', 'month', 'day', 'hour', 
            'PV_q10', 'PV_q20', 'PV_q30', 'PV_q40', 'PV_q50', 'PV_q60', 'PV_q70', 'PV_q80', 'PV_q90', 'PV_actual', 
            'wind_q10', 'wind_q20', 'wind_q30', 'wind_q40', 'wind_q50', 'wind_q60', 'wind_q70', 'wind_q80', 'wind_q90', 'wind_actual',
            'MIP_q10', 'MIP_q20', 'MIP_q30', 'MIP_q40', 'MIP_q50', 'MIP_q60', 'MIP_q70', 'MIP_q80', 'MIP_q90','MIP_actual',
            'SSP_q10', 'SSP_q20', 'SSP_q30', 'SSP_q40', 'SSP_q50', 'SSP_q60', 'SSP_q70', 'SSP_q80', 'SSP_q90','SSP_actual',
            'charge/discharge_bid', 'charge/discharge_plan', 'charge/discharge_actual',
            'SoC_bid', 'SoC_plan', 'SoC_actual', 'energytransfer_bid', 'energytransfer_plan', 'energytransfer_actual',
            'energy_profit', 'imbalance_penalty', 'total_profit'
        ]
        # 空のDataframeを作成
        self.df_result = pd.DataFrame(columns=col)

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


    def get_data(self,idx):
        return  self.datafame(idx,:)


    def get_result_csv(self):
        # 表示
        print("dataframe作成")
        # DataframeをCSVファイルとして出力
        # 作成時間も同時に記入できるとよさそう？
        self.df_result.to_csv('./Battery-Control-By-Reinforcement-Learning/result_dataframe.csv', index=False)