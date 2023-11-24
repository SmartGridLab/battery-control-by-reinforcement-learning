import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# データを読み込む
df = pd.read_csv('Energy_Data_02.csv')

# 新しい 'time' 列を作成
df['time'] = pd.to_datetime(df['dtm'])
# 年、月、日、時間を抽出
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour + df['time'].dt.minute / 60

# 時間に関する周期的特徴を計算
df['hourCos'] = np.cos(df['hour'] * 2 * np.pi / 24)
df['hourSin'] = np.sin(df['hour'] * 2 * np.pi / 24)

# 24時間前の 'SS_Price' を新しい列として追加（48期前）
df['SS_Price_24h_ago'] = df['SS_Price'].shift(48)
# 前方の値でNaNを補完する
df['SS_Price_24h_ago'].fillna(method='ffill', inplace=True)
# 48時間前の 'SS_Price' を新しい列として追加（96期前）
df['SS_Price_48h_ago'] = df['SS_Price'].shift(96)
df['SS_Price_48h_ago'].fillna(method='ffill', inplace=True)
# 1週間前（168時間前）の 'SS_Price' を新しい列として追加（336期前）
df['SS_Price_1week_ago'] = df['SS_Price'].shift(336)
df['SS_Price_1week_ago'].fillna(method='ffill', inplace=True)

# 「DA_Price」に対しても時間遅れの特徴量を作成
df['DA_Price_24h_ago'] = df['DA_Price'].shift(48)
df['DA_Price_24h_ago'].fillna(method='bfill', inplace=True)
df['DA_Price_48h_ago'] = df['DA_Price'].shift(96)
df['DA_Price_48h_ago'].fillna(method='bfill', inplace=True)
df['DA_Price_1week_ago'] = df['DA_Price'].shift(336)
df['DA_Price_1week_ago'].fillna(method='bfill', inplace=True)

df['Solar_MW_24h_ago'] = df['Solar_MW'].shift(48)
df['Solar_MW_24h_ago'].fillna(method='bfill', inplace=True)

df['Wind_MW_24h_ago'] = df['Wind_MW'].shift(48)
df['Wind_MW_24h_ago'].fillna(method='bfill', inplace=True)


# データをトレーニングとテストセットに分割（上からの順番で）
train_df = df.iloc[:int(0.8*len(df))].copy()
test_df = df.iloc[int(0.8*len(df)):].copy()

# 線形補間によりNaNを置換する
test_df['DA_Price'].interpolate(method='linear', inplace=True)
# 線形補間によりNaNを置換する
test_df['Solar_MW'].interpolate(method='linear', inplace=True)
# 線形補間によりNaNを置換する
test_df['Wind_MW'].interpolate(method='linear', inplace=True)


def pinball_loss(y_true, y_pred, tau=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_true - y_pred
    loss = np.maximum(tau * diff, (tau - 1) * diff)
    return loss

# 量子ごとのピンボール損失を保存する辞書を作成
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#「SSP」開始
quantile_losses_SSP = {q: [] for q in quantiles}

# 特徴量リスト
features_SSP = ['SS_Price_24h_ago', 'SS_Price_48h_ago', 'SS_Price_1week_ago', 'Solar_MW', 'Wind_MW', 'hourSin', 'hourCos']

# 量子回帰モデルを特徴量でフィットさせる
quantile_models_SSP = {}
for q in quantiles:
    formula = f'SS_Price ~ {" + ".join(features_SSP)}'
    quantile_models_SSP[q] = smf.quantreg(formula, train_df).fit(q=q)

# 空のDataFrameを作成
predictions_df_SSP = pd.DataFrame(index=test_df.index)

# predictions_dfに年、月、日、時間の列を追加
predictions_df_SSP['time'] = test_df['time']
predictions_df_SSP['year'] = test_df['time'].dt.year
predictions_df_SSP['month'] = test_df['time'].dt.month
predictions_df_SSP['day'] = test_df['time'].dt.day
predictions_df_SSP['hour'] = test_df['time'].dt.hour + test_df['time'].dt.minute / 60

for q in quantiles:
    model = quantile_models_SSP[q]
    pred = model.predict(test_df[features_SSP])
    predictions_df_SSP[f'SSP_q_{q}'] = pred
    loss = pinball_loss(test_df['SS_Price'], pred, tau=q)
    quantile_losses_SSP[q].extend(loss)  # extendを使用してリストに要素を追加

# 予測結果と実際のSS_Priceを含むDataFrameを作成します。
predictions_df_SSP['SS_Price_actual'] = test_df['SS_Price']

# 平均ピンボール損失を計算して印刷（SS_Price用）
for q, losses in quantile_losses_SSP.items():  # 変数名を更新
    average_loss = np.mean(losses)
    print(f"Average Pinball Loss for SS_Price Quantile {q}: {average_loss}")


# 「DA_Price」開始
quantile_losses_DA_Price = {q: [] for q in quantiles}

# 特徴量リスト
features_DA_Price = ['DA_Price_24h_ago', 'DA_Price_1week_ago', 'hourSin', 'hourCos']

# 量子回帰モデルを特徴量でフィットさせるdf['DA_Price_24h_ago'] = df['DA_Price'].shift(48)
df['DA_Price_24h_ago'].fillna(method='bfill', inplace=True)

quantile_models_DA_Price = {}  # 名前を変更
for q in quantiles:
    formula = f'DA_Price ~ {" + ".join(features_DA_Price)}'
    quantile_models_DA_Price[q] = smf.quantreg(formula, train_df).fit(q=q)

# 「DA_Price」の予測結果を保存するためのDataFrameを作成
predictions_df_DA_Price = pd.DataFrame(index=test_df.index)
predictions_df_DA_Price['time'] = test_df['time']
predictions_df_DA_Price['year'] = test_df['time'].dt.year
predictions_df_DA_Price['month'] = test_df['time'].dt.month
predictions_df_DA_Price['day'] = test_df['time'].dt.day
predictions_df_DA_Price['hour'] = test_df['time'].dt.hour + test_df['time'].dt.minute / 60

for q in quantiles:
    model_DA_Price = quantile_models_DA_Price[q]
    pred_DA_Price = model_DA_Price.predict(test_df[features_DA_Price])
    predictions_df_DA_Price[f'DA_Price_q_{q}'] = pred_DA_Price
    loss_DA_Price = pinball_loss(test_df['DA_Price'], pred_DA_Price, tau=q)
    quantile_losses_DA_Price[q].extend(loss_DA_Price)

# 実際の「DA_Price」をDataFrameに追加
predictions_df_DA_Price['DA_Price_actual'] = test_df['DA_Price']

# 「SS_Price」の予測結果を含むDataFrameと結合
predictions_df_final = predictions_df_SSP.join(predictions_df_DA_Price.drop(columns=['time', 'year', 'month', 'day', 'hour']))

# 修正されたDataFrameをCSVファイルに保存
predictions_df_final.to_csv('predictions_actuals.csv', mode='w')

# 「DA_Price」の平均ピンボール損失を計算して印刷
for q, losses in quantile_losses_DA_Price.items():
    average_loss_DA_Price = np.mean(losses)
    print(f"Average Pinball Loss for DA_Price Quantile {q}: {average_loss_DA_Price}")

# 特定の日付を選択（例：2023年3月15日）
selected_date = '2023-03-16'

# その日付に対応する行をフィルタリング
selected_df = predictions_df_final[predictions_df_final['time'].dt.date == pd.to_datetime(selected_date).date()]

# Matplotlibでグラフ描画
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)

# SSPのグラフ
axes[0].plot(selected_df['time'], selected_df['SS_Price_actual'], label='SS_Price_actual', color='black', linewidth=2)
for q in quantiles:
    axes[0].plot(selected_df['time'], selected_df[f'SSP_q_{q}'], label=f'SSP_q_{q}')

axes[0].set_title('SS_Price Predictions vs Actual')
axes[0].legend()
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[0].set_ylabel('SS_Price')

# DA_Priceのグラフ
axes[1].plot(selected_df['time'], selected_df['DA_Price_actual'], label='DA_Price_actual', color='black', linewidth=2)
for q in quantiles:
    axes[1].plot(selected_df['time'], selected_df[f'DA_Price_q_{q}'], label=f'DA_Price_q_{q}')

axes[1].set_title('DA_Price Predictions vs Actual')
axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[1].set_ylabel('DA_Price')

# X軸の設定
for ax in axes:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.grid(True)

plt.tight_layout()
plt.show()