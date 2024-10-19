# 充放電計画の性能評価のためのデータを集めるコード
# - PV発電量の実績値、電力価格の実績値、不平衡電力価格の実績値を取得する
# - 実績値ベースでの売電による収益の計算を行う

import pandas as pd

def main():
    print("---実績データ参照開始---")

    #current_date.csvよりyear, month, dayを取得(充放電計画を行った)
    date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
    # date_infoは {'year': year, 'month': month, 'day': day} の形式
    date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
    latest_date = date_info['date'].max()

    year = latest_date.year
    month = latest_date.month
    day = latest_date.day
    
    # 全てのデータを読み込み
    original_df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

    # 最新日付のデータをフィルタリング
    df = original_df[(original_df['year'] == year) & 
                     (original_df['month'] == month) & 
                     (original_df['day'] == day)]
    # デバッグ用：フィルタリング後の行数を出力
    print(f"Filtered rows for date {year}-{month}-{day}: {len(df)}")

    
    # 現在日付のPV, price, imbalance実績値を取得
    data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
    PV_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["PVout"]]
    energyprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["price",]]
    imbalanceprice_actual = data[(data["year"] == year) & (data["month"] == month) & (data["day"] == day)][["imbalance"]]

    # pd -> numpy変換
    PV_actual = PV_actual.values.flatten()

    energyprice_actual = energyprice_actual.values.flatten()
    imbalanceprice_actual = imbalanceprice_actual.values.flatten()


    ### 実績データの挿入
    
    # i+1行目からi+48行目に実績データを挿入
    df.loc[df.index[:48], 'PV_actual[kW]'] = PV_actual
    df.loc[df.index[:48], 'energyprice_actual[Yen/kWh]'] = energyprice_actual * 0.5
    df.loc[df.index[:48], 'imbalanceprice_actual[Yen/kWh]'] = imbalanceprice_actual * 0.5
   
    # フィルタリングした部分のデータを元データから消す
    original_df_erase = original_df[~((original_df['year'] == year) & 
                             (original_df['month'] == month) & 
                             (original_df['day'] == day))]
    
    print(df)
    print(original_df_erase)
   # 元のデータフレームを更新
    original_df_concat = pd.concat([original_df_erase, df], axis=0)
  
    # result_dataframe.csvを保存
    original_df_concat.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", index=False)

    print("---実績データ書き込み完了---")

if __name__ == "__main__":
    main()

