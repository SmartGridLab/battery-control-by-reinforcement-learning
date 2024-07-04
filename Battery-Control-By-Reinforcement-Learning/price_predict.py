import pandas as pd
import numpy as np
import math as ma
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

#スタート
print("\n\n---電力価格予測プログラム開始---\n\n")

# データの読み込み
# 学習データ
input_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")


date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        # date_infoは {'year': year, 'month': month, 'day': day} の形式
date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
latest_date = date_info['date'].max()

year = latest_date.year
month = latest_date.month
day = latest_date.day

# 日付でフィルタリング
price_predict = input_data[(input_data['year'] == year) & 
                            (input_data['month'] == month) & 
                            (input_data['day'] == day) ]

# 使用するパラメータ
#parameters = ['temperature', 'total precipitation', 'u-component of wind', 'v-component of wind',
              #'radiation flux', 'pressure', 'relative humidity', 'hourSin', 'hourCos', 'PVout']
parameters = ['radiation flux', 'PVout', 'temperature', 'hourCos']           
predict_parameters = ['price', 'imbalance']


# データの前処理
scaler = MinMaxScaler()
input_data[parameters] = scaler.fit_transform(input_data[parameters])
price_predict[parameters] = scaler.transform(price_predict[parameters])

# 学習データとターゲットデータの作成
X = input_data[parameters].values
y = input_data[predict_parameters].values

# モデルの定義
hidden_units = [64, 64, 64]  # 隠れ層のユニット数
epochs = 100  # エポック数

model = keras.Sequential()
model.add(keras.layers.Dense(hidden_units[0], activation='relu', input_shape=(len(parameters),)))
for units in hidden_units[1:]:
    model.add(keras.layers.Dense(units, activation='relu'))
model.add(keras.layers.Dense(len(predict_parameters)))

# モデルのコンパイル
model.compile(optimizer='adam', loss='mse')

# モデルの学習
model.fit(X, y, epochs=epochs, verbose=0)

# 予測の実行
predictions = model.predict(price_predict[parameters].values)

# 予測結果の保存
pred_df = pd.DataFrame(columns=["year","month","day","hour","hourSin","hourCos","PVout","price","imbalance"])

pred_df["price"] = predictions[:, 0]  # "price" の予測値を代入
pred_df["imbalance"] = predictions[:, 1]  # "imbalance" の予測値を代入
pred_df["year"] = price_predict["year"].values
pred_df["month"] = price_predict["month"].values
pred_df["day"] = price_predict["day"].values
pred_df["hour"] = price_predict["hour"].values
pred_df["hourSin"] = price_predict["hourSin"].values
pred_df["hourCos"] = price_predict["hourCos"].values
#pred_df["upper"] = price_predict["upper"].values
#pred_df["lower"] = price_predict["lower"].values
pred_df["PVout"] = price_predict["PVout"].values
pred_df.to_csv("Battery-Control-By-Reinforcement-Learning/price_predict.csv", index=False)

# グラフの描画
#plt.figure(figsize=(10, 5))
#plt.plot(price_predict['hour'], predictions[:, 0], label='price')
#plt.plot(price_predict['hour'], predictions[:, 1], label='imbalance')
#plt.xlabel('hour')
#plt.ylabel('value')
#plt.title('Price and Imbalance Prediction')
#plt.legend()
#plt.show()

#終了
print("\n\n---電力価格予測プログラム終了---")