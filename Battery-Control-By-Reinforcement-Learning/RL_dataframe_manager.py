# csvからのデータの読み込みを行うクラス

from email import header
import pandas as pd
import os

class Dataframe_Manager(): 
    ## 強化学習の学習に使うテーブル(df_train)を作成
    def get_train_df(self):
        # CSVファイル(input_data2022.csv)から学習データを読み込む
        # 読み込む行を列名で指定：year,month,day,hour, PVout, price, imbalance
        # df_traindata = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv",
        #                               usecols=["year","month","day","hour","PVout","price","imbalance"])        
        df_traindata = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022_all_0.csv",
                                      usecols=["year","month","day","hour","PVout","price","imbalance"])        
        
        return df_traindata

    def get_test_df(self):

        date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        # date_infoは {'year': year, 'month': month, 'day': day} の形式
        date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
        latest_date = date_info['date'].max()

        year = latest_date.year
        month = latest_date.month
        day = latest_date.day

        # - 前提：PV発電予測と価格予測の結果のcsvがあること
        # - CSVファイルから予測結果のデータを読み込む
        # 電力価格データからyear, month, day,hour,price, imbalanceを読み込む
        # price_predict = pd.read_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv", 
        #                             usecols=["year","month","day","hour","price","imbalance"])
        price_predict = pd.read_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv", 
                                    usecols=["year","month","day","hour","price","imbalance"])
        
        # 日付でフィルタリング
        price_predict = price_predict[(price_predict['year'] == year) & 
                                      (price_predict['month'] == month) & 
                                      (price_predict['day'] == day) ]


        # 列名を変更する
        price_predict = price_predict.rename(columns={'price': 'energyprice_predict_bid[Yen/kWh]', 'imbalance': 'imbalanceprice_predict_bid[Yen/kWh]'})
        # # PV予測結果データからyear, month, day,hour, PVoutを読み込む
        # pv_predict = pd.read_csv("Battery-Control-By-Reinforcement-Learning/pv_predict.csv",
        #                             usecols=["year","month","day","hour","PVout"])
        # PV予測結果データからyear, month, day,hour, PVoutを読み込む
        pv_predict = pd.read_csv("Battery-Control-By-Reinforcement-Learning/pv_predict.csv",
                                    usecols=["year","month","day","hour","PVout"])
        
        # 日付でフィルタリング
        pv_predict = pv_predict[(pv_predict['year'] == year) & 
                                (pv_predict['month'] == month) & 
                                (pv_predict['day'] == day) ]

        # 列名を変更する
        pv_predict = pv_predict.rename(columns={'PVout': 'PV_predict_bid[kW]'})
        # price_predictとpv_predictを結合（キーはyear,month,day,hourが全て一致） 
        df_testdata = pd.merge(price_predict, pv_predict, how='outer', on=['year','month','day','hour'])
        # "SoC_bid", "charge/discharge_bid"を列名として追加。行数はdf_testdataの行数と同じで、全て-999を入れる
        # -999は、欠損値を表す（NaNと同じ）
        df_testdata["SoC_bid[%]"] = [-999 for i in range(len(df_testdata))]
        df_testdata["charge/discharge_bid[kWh]"] = [-999 for i in range(len(df_testdata))]
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
        # SoC_bid: [計画値] 充放電計画策定時(charge/discharge_bid)のSoC [%]
        # SoC_realtime: [計画値] 実行30分前の充放電計画策定時(charge/discharge_realtime)のSoC [%]
        # SoC_actual_bid: [実績値] 充放電実行結果のSoC(charge/discharge_actual_bidへできるだけ近づけた制御を実施) [%]
        # SoC_actual_realtime: [実績値] 充放電実行結果のSoC(charge/discharge_actual_realtimeへできるだけ近づけた制御を実施) [%]
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
            'PV_predict_bid[kW]', 'PV_predict_realtime[kW]', 'PV_actual[kW]',
            # 電力価格関連
            'energyprice_predict_bid[Yen/kWh]', 'energyprice_predict_realtime[Yen/kWh]', 'energyprice_actual[Yen/kWh]', 
            'imbalanceprice_predict_bid[Yen/kWh]', 'imbalanceprice_predict_realtime[Yen/kWh]', 'imbalanceprice_actual[Yen/kWh]',
            # 充放電量関連
            'charge/discharge_bid[kWh]', 'charge/discharge_realtime[kWh]', 'charge/discharge_actual_bid[kWh]', 'charge/discharge_actual_realtime[kWh]',
            # SoC関連
            'SoC_bid[%]', 'SoC_realtime[%]', 'SoC_actual_bid[%]', 'SoC_actual_realtime[%]', 
            # 売電量関連
            'energytransfer_bid[kWh]', 'energytransfer_realtime[kWh]', 'energytransfer_actual_bid[kWh]','energytransfer_actual_realtime[kWh]',
            # 売電利益関連
            # - imbalancepenalty_bidは存在しない。bidの段階ではimbalanceは発生しないので。
            # - imbalancepenalty_realtimeは存在しえる。realtimeでのスケジュール策定時にimbalanceが避けられない場合が想定されるので。
            'energyprofit_bid[Yen]', 'energyprofit_realtime[Yen]', 
            'imbalancepenalty_realtime[Yen]', 'imbalancepenalty_actual_bid[Yen]', 'imbalancepenalty_actual_realtime[Yen]', 
            'totalprofit_bid[Yen]', 'totalprofit_realtime[Yen]', 'totalprofit_actual_bid[Yen]', 'totalprofit_actual_realtime[Yen]',
            # 動作モード：operateの条件分岐を見るためのもの。デバッグ用
            'mode_bid', 'mode_realtime'
        ]

        # 空のDataframeを作成
        dataframe = pd.DataFrame(columns=col)
        # 空のDataframeにget_test_df()の結果を追加
        dfmanager = Dataframe_Manager()
        dataframe = dataframe.append(dfmanager.get_test_df(), ignore_index=True)

        return dataframe

        