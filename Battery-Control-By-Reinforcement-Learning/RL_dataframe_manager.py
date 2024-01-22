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
        price_predict = price_predict.rename(columns={'price': 'energyprice_predict_bid', 'imbalance': 'imbalanceprice_predict_bid'})
        # PV予測結果データからyear, month, day,hour, PVoutを読み込む
        pv_predict = pd.read_csv("Battery-Control-By-Reinforcement-Learning/pv_predict.csv",
                                    usecols=["year","month","day","hour","PVout"])
        # 列名を変更する
        pv_predict = pv_predict.rename(columns={'PVout': 'PV_predict_bid'})
        # price_predictとpv_predictを結合（キーはyear,month,day,hourが全て一致） 
        df_testdata = pd.merge(price_predict, pv_predict, how='outer', on=['year','month','day','hour'])
        return df_testdata


    ## 強化学習の結果を入れるテーブル(df_result)を作成
    def get_resultform_df(self):
        # 列名をリストとして定義
        # year: 2020, 2021
        # month: 
        # day:
        # hour: 0~23.5の30分刻み48コマ
        # PV_predict_bid: PV発電量の予測値(入札時) [kW]
        # PV_predict_realtime: PV発電量の予測値(30分前予測値) [kW]
        # PV_actual: PV発電量の実績値 [kW]
        # energyprice_predict_bid: 電力価格の予測値(入札時) [円/kWh]
        # energyprice_predict_realtime: 電力価格の予測値(30分前予測値) [円/kWh]
        # energyprice_actual: 電力価格の実績値 [円/kWh]
        # imbalanceprice_predict_bid: インバランス価格の予測値(入札時) [円/kWh]
        # imbalanceprice_predict_realtime: インバランス価格の予測値(30分前予測値) [円/kWh]
        # imbalanceprice_actual: インバランス価格の実績値 [円/kWh]
        # charge/discharge_bid: 充放電量の計画値(入札時) [kWh]
        # charge/discharge_realtime: 充放電量の計画値(実行30分前のPV予測の結果を加味して、策定し直したもの) [kWh]
        # charge/discharge_actual_bid: 充放電量の実績値(入札時の計画charge/discharge_bidへできるだけ近づくように充放電した結果) [kWh]
        # charge/discharge_actual_realtime: 充放電量の実績値(実行30分前の計画charge/discharge_realtimeへできるだけ近づくように充放電した結果）[kWh]
        # SoC_bid: [計画値] 充放電計画策定時(charge/discharge_bid)のSoC [kWh]
        # SoC_realtime: [計画値] 実行30分前の充放電計画策定時(charge/discharge_realtime)のSoC [kWh]
        # SoC_actual_bid: [実績値] 充放電実行結果のSoC(charge/discharge_actual_bidへできるだけ近づけた制御を実施) [kWh]
        # SoC_actual_realtime: [実績値] 充放電実行結果のSoC(charge/discharge_actual_realtimeへできるだけ近づけた制御を実施) [kWh]
        # energytransfer_bid: [計画値] 充放電計画策定時(charge/discharge_bid)のPVからの直接売電を含めた売電量 [kWh]
        # energytransfer_realtime: [計画値] 実行30分前の充放電計画策定時(charge/discharge_realtime)のPVからの直接売電を含めた売電量 [kWh]
        # energytransfer_actual_bid: [実績値] 充放電実行結果の売電量(charge/discharge_actual_bidの制御実績による売電) [kWh]
        # energytransfer_actual_realtime: [実績値] 充放電実行結果の売電量(charge/discharge_actual_realtimeの制御実績による売電) [kWh]
        # energyprofit_bid: [計画値] 充放電計画策定時(charge/discharge_bid)のPVからの直接売電を含めた売電利益 [円]
        # energyprofit_realtime: [計画値] 実行30分前の充放電計画策定時(charge/discharge_realtime)のPVからの直接売電を含めた売電利益 [円]
        # !!! imbalancepenalty_bid: [予測値] 充放電計画策定時(charge/discharge_bid)のインバランスペナルティ予測値[円]
        # - imbalancepenalty_bidは存在しない（bidの段階ではimbalanceは発生しえない）
        # imbalancepenalty_realtime: [予測値] 実行30分前の充放電計画策定時(charge/discharge_realtime)のインバランスペナルティ予測値[円]
        # imbalancepenalty_actual_bid: [実績値] 充放電実行結果のインバランスペナルティ(charge/discharge_actual_bidの制御実績によるインバランスペナルティ) [円]
        # imbalancepenalty_actual_realtime: [実績値] 充放電実行結果のインバランスペナルティ(charge/discharge_actual_realtimeの制御実績によるインバランスペナルティ) [円]
        # totalprofit_bid: [計画値] 充放電計画策定時(charge/discharge_bid)の売電利益とインバランスペナルティの合計 [円]
        # totalprofit_realtime: [計画値] 実行30分前の充放電計画策定時(charge/discharge_realtime)の売電利益とインバランスペナルティの合計 [円]
        # totalprofit_actual_bid: [実績値] 充放電実行結果の売電利益とインバランスペナルティの合計(charge/discharge_actual_bidの制御実績による売電利益とインバランスペナルティの合計) [円]
        # totalprofit_actual_realtime: [実績値] 充放電実行結果の売電利益とインバランスペナルティの合計(charge/discharge_actual_realtimeの制御実績による売電利益とインバランスペナルティの合計) [円]
        # mode: [計画値] 充放電計画策定時(charge/discharge_bid)の動作モード。0:放電, 1:充電, 2:待機
        # mode_realtime: [計画値] 実行30分前の充放電計画策定時(charge/discharge_realtime)の動作モード。0:放電, 1:充電, 2:待機

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
            # - imbalancepenalty_bidは存在しない。bidの段階ではimbalanceは発生しないので。
            # - imbalancepenalty_realtimeは存在しえる。realtimeでのスケジュール策定時にimbalanceが避けられない場合が想定されるので。
            'energyprofit_bid', 'energyprofit_realtime', 
            'imbalancepenalty_realtime', 'imbalancepenalty_actual_bid', 'imbalancepenalty_actual_realtime', 
            'totalprofit_bid', 'totalprofit_realtime', 'totalprofit_actual_bid', 'totalprofit_actual_realtime',
            # 動作モード：operateの条件分岐を見るためのもの。デバッグ用
            'mode_bid', 'mode_realtime'
        ]

        # 空のDataframeを作成
        dataframe = pd.DataFrame(columns=col)
        # 空のDataframeにget_test_df()の結果を追加
        dfmanager = Dataframe_Manager()
        dataframe = dataframe.append(dfmanager.get_test_df(), ignore_index=True)

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
        self.trainDataTable = get_PriceImbPVout(input_data)
        self.predictorDataTable = get_PriceImbPVout(predict_data)

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

    def write_testresult_csv(self, data):
        # result_dataframe.csvを読み込む
        # もしresult_dataframe.csvがなければ、get_resultform_df()で作成する
        try:
            df_result = pd.read_csv('./Battery-Control-By-Reinforcement-Learning/result_dataframe.csv')
        except:
            df_result = self.get_resultform_df()
        
        # dataの'SoC_bid'をdf_resultの'SoC_bid'に追加
        df_result['SoC_bid'] = data['SoC_bid']
        df_result['charge/discharge_bid'] = data['charge/discharge_bid']
        df_result.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)        
        