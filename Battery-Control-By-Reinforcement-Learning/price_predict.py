import pandas as pd
import numpy as np
import math as ma
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from result_inputdata_reference import ResultInputDataReference as RIRD

class PricePredict:
    def __init__(self):
        self.RIRD = RIRD()
        self.input_data = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
        self.year, self.month, self.day = self.RIRD.get_current_date()
        self.price_predict = self.input_data[(self.input_data['year'] == self.year) &
                                            (self.input_data['month'] == self.month) &
                                            (self.input_data['day'] == self.day)]
        
        # 使用するパラメータ
        # parameters = ['temperature', 'total precipitation', 'u-component of wind', 'v-component of wind',
                    #'radiation flux', 'pressure', 'relative humidity', 'hourSin', 'hourCos', 'PVout']
        self.parameters = ['radiation flux', 'PVout', 'temperature', 'hourCos']           
        self.predict_parameters = ['price', 'imbalance']

    def predict_values(self, mode): 
        print("\n---電力価格予測プログラム開始---\n")

        # データの前処理
        scaler = MinMaxScaler()
        self.input_data[self.parameters] = scaler.fit_transform(self.input_data[self.parameters])
        self.price_predict[self.parameters] = scaler.transform(self.price_predict[self.parameters])

        # 学習データとターゲットデータの作成
        X = self.input_data[self.parameters].values
        y = self.input_data[self.predict_parameters].values

        # モデルの定義
        hidden_units = [64, 64, 64]  # 隠れ層のユニット数
        epochs = 100  # エポック数

        model = keras.Sequential()
        model.add(keras.layers.Dense(hidden_units[0], activation='relu', input_shape=(len(self.parameters),)))
        for units in hidden_units[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dense(len(self.predict_parameters)))

        # モデルのコンパイル
        model.compile(optimizer='adam', loss='mse')

        # モデルの学習
        model.fit(X, y, epochs=epochs, verbose=0)

        # 予測の実行
        predictions = model.predict(self.price_predict[self.parameters].values)

        # 予測結果の保存
        pred_df = pd.DataFrame(columns=["year","month","day","hour","hourSin","hourCos","PVout","price","imbalance"])

        pred_df["price"] = predictions[:, 0]  # "price" の予測値を代入
        pred_df["imbalance"] = predictions[:, 1]  # "imbalance" の予測値を代入
        pred_df["year"] = self.price_predict["year"].values
        pred_df["month"] = self.price_predict["month"].values
        pred_df["day"] = self.price_predict["day"].values
        pred_df["hour"] = self.price_predict["hour"].values
        pred_df["hourSin"] = self.price_predict["hourSin"].values
        pred_df["hourCos"] = self.price_predict["hourCos"].values
        #pred_df["upper"] = self.price_predict["upper"].values
        #pred_df["lower"] = self.price_predict["lower"].values
        # pv_predict.pyのPV予測よりもprice_predict.csvの予測の方が正確そう
        pred_df["PVout"] = self.price_predict["PVout"].values

        #---------------------- 人工テストデータの作成 ----------------------#
        # if mode == "bid":
        #     pred_df["price"] = (self.RIRD.energyprice_actual) * 2.0
        #     pred_df["imbalance"] = (self.RIRD.imbalanceprice_actual) * 2.0
        # elif mode == "realtime":
        #     pred_df["price"] = self.RIRD.energyprice_actual
        #     pred_df["imbalance"] = self.RIRD.imbalanceprice_actual
        #---------------------- 人工テストデータの作成 ----------------------#
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
        print("\n---電力価格予測プログラム終了---")