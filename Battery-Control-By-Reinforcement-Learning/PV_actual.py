#このプログラムはシミュレーションにおいて実際のPV発電量を仮想的に出力させるプログラムです

import pandas as pd
import numpy as np

def calculate_solar_power(input_file):
    # データの読み込み
    data = pd.read_csv(input_file)

    # 最初の1行を取得
    first_row = data.head(1)

    # 定数の定義
    RATED_POWER = 2.0  # kW
    CONVERSION_EFFICIENCY = 0.15  # 変換効率
    RADIATION_WEIGHT = 0.8  # 日射量の重み
    TEMPERATURE_COEFFICIENT = -0.2  # 温度係数
    TEMPERATURE_REF = 25  # 参照温度
    WIND_SPEED_COEFFICIENT = 0.01  # 風速係数

    # 太陽光発電量の計算
    def calc_power(row):
        # 日射量が0の場合は発電量を0にする
        if row['global_horizontal_radiation'] == 0:
            return 0

        # 気象条件から発電効率を計算する
        temperature_effect = (row['temperature'] - TEMPERATURE_REF) * TEMPERATURE_COEFFICIENT  # 温度係数
        wind_speed_effect = np.sqrt(row['u-component_of_wind'] ** 2 + row['v-component_of_wind'] ** 2) * WIND_SPEED_COEFFICIENT

        # 日時の要素による変動を加える
        month_effect = np.cos((row['month'] - 1) * 2 * np.pi / 12)  # 月の影響
        hour_effect = np.cos((row['hour'] - 12) * 2 * np.pi / 24)  # 時間の影響

        efficiency = CONVERSION_EFFICIENCY * (1 + RADIATION_WEIGHT * row['global_horizontal_radiation'] + temperature_effect + wind_speed_effect + month_effect + hour_effect)

        # 定格出力を超えないようにする
        power = RATED_POWER * min(efficiency, 1) * row['global_horizontal_radiation'] / 1000

        return power

    # 最初の1行の発電量を計算
    result = first_row.apply(calc_power, axis=1)

    return result
