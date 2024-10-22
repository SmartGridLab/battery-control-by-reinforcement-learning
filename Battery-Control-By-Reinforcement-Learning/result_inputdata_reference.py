# 充放電計画の性能評価のためのデータを集めるコード
# - PV発電量の実績値、電力価格の実績値、不平衡電力価格の実績値を取得する
# - 実績値ベースでの売電による収益の計算を行う

import pandas as pd

def main():
    print("---実績データ参照開始---")
    # 現在日付を取得
    date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
    # date_infoは {'year': year, 'month': month, 'day': day, 'hour': hour} の形式
    date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
    latest_date = date_info['date'].max()
    year = latest_date.year
    month = latest_date.month
    day = latest_date.day
    # 全てのデータを読み込み
    df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
    # 現在日付のデータをフィルタリング
    df = df_original[(df_original['year'] == year) & 
                     (df_original['month'] == month) & 
                     (df_original['day'] == day)].reset_index(drop = True)
    # 実績データがすでに存在するか確認（存在する ＝ True, 存在しない = False）
    actualdata_exists = ((df['year'] == year) & 
                         (df['month'] == month) & 
                         (df['day'] == day) &
                         pd.notna(df['PV_actual[kW]']) &
                         pd.notna(df['energyprice_actual[Yen/kWh]']) &
                         pd.notna(df['imbalanceprice_actual[Yen/kWh]'])).any()
    if actualdata_exists:
        print("---実績データがすでに存在します---")
    else:
        print("---実績データが存在しないため、実績データを入力します---")
        data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
        PV_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["PVout"]]
        energyprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["price"]]
        imbalanceprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["imbalance"]]
        # pd -> numpy変換(2次元配列 → 1次元配列)
        PV_actual = PV_actual.values.flatten()
        energyprice_actual = energyprice_actual.values.flatten()
        imbalanceprice_actual = imbalanceprice_actual.values.flatten()
        # 実績データの挿入
        df.loc[df.index[:48], 'PV_actual[kW]'] = PV_actual
        df.loc[df.index[:48], 'energyprice_actual[Yen/kWh]'] = energyprice_actual * 0.5
        df.loc[df.index[:48], 'imbalanceprice_actual[Yen/kWh]'] = imbalanceprice_actual * 0.5
        # year, month, day, hourのをindexとして設定
        df_original.set_index(['year', 'month', 'day', 'hour'], inplace = True)
        df.set_index(['year', 'month', 'day', 'hour'], inplace = True)
        df_original.update(df)
        # indexを振りなおす
        df_original.reset_index(inplace = True)
        df.reset_index(inplace = True)
        df_original.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", header = True, index=False)
        print("---実績データの入力完了---")

if __name__ == "__main__":
    main()

