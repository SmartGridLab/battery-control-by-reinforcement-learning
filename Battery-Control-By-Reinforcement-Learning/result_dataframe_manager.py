import pandas as pd

# 表示
print("dataframe作成")

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
# imbalancepenalty_bid: [予測値] 充放電計画策定時(charge/discharge_bid)のインバランスペナルティ予測値[円/kWh]
# imbalancepenalty_realtime: [予測値] 実行30分前の充放電計画策定時(charge/discharge_realtime)のインバランスペナルティ予測値[円/kWh]
# imbalancepenalty_actual_bid: [実績値] 充放電実行結果のインバランスペナルティ(charge/discharge_actual_bidの制御実績によるインバランスペナルティ) [円]
# imbalancepenalty_actual_realtime: [実績値] 充放電実行結果のインバランスペナルティ(charge/discharge_actual_realtimeの制御実績によるインバランスペナルティ) [円]
# totalprofit_bid: [計画値] 充放電計画策定時(charge/discharge_bid)の売電利益とインバランスペナルティの合計 [円]
# totalprofit_realtime: [計画値] 実行30分前の充放電計画策定時(charge/discharge_realtime)の売電利益とインバランスペナルティの合計 [円]
# totalprofit_actual_bid: [実績値] 充放電実行結果の売電利益とインバランスペナルティの合計(charge/discharge_actual_bidの制御実績による売電利益とインバランスペナルティの合計) [円]
# totalprofit_actual_realtime: [実績値] 充放電実行結果の売電利益とインバランスペナルティの合計(charge/discharge_actual_realtimeの制御実績による売電利益とインバランスペナルティの合計) [円]
# mode: [計画値] 充放電計画策定時(charge/discharge_bid)の動作モード。0:放電, 1:充電, 2:待機
# mode_realtime: [計画値] 実行30分前の充放電計画策定時(charge/discharge_realtime)の動作モード。0:放電, 1:充電, 2:待機


columns = [
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
    'energyprofit_bid', 'energyprofit_realtime', 
    'imbalancepenalty_bid', 'imbalancepenalty_realtime', 'imbalancepenalty_actual_bid', 'imbalancepenalty_actual_realtime', 
    'totalprofit_bid', 'totalprofit_realtime', 'totalprofit_actual_bid', 'totalprofit_actual_realtime',
    # 動作モード：operateの条件分岐を見るためのもの。デバッグ用
    'mode', 'mode_realtime'
]

# 空のDataframeを作成
dataframe = pd.DataFrame(columns=columns)

# DataframeをCSVファイルとして出力
# 作成時間も同時に記入できるとよさそう？
dataframe.to_csv('./Battery-Control-By-Reinforcement-Learning/result_dataframe.csv', index=False)