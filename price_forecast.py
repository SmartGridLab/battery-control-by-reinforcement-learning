import tensorflow as tf
from tensorflow import keras
import numpy as py
import pandas as pd
import math as ma
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from tensorflow import keras

##グラフ作成
def graph(time_stamp, pred, target, name):
    Figure = plt.figure()
    graph_legend  = "Predict(" + name + ")"
    plt.plot(time_stamp, pred, "blue")
    plt.plot(time_stamp, target, "red")
    plt.legend([graph_legend, "target"]) # 凡例
    plt.xlabel("time [hour]") # 横軸
    plt.ylabel("price [yen]") # 縦軸
    plt.close()
    
    return Figure
        
## RMSE
def RMSE(pred, Predict, name):
    new_name = "Score(" + name + "): %.2f RMSE"
    testScore = ma.sqrt(mean_squared_error(pred[:,0], Predict[:,0]))
    return testScore

##データのロード
data_name = "dataset_2019-2021" # dataset_2019-2021
load_name = data_name + ".csv" 
PV_data = pd.read_csv(load_name)
sin_data = py.sin(PV_data["hour"]/12*(ma.pi)) # 時系列データをsinに変換
cos_data = py.cos(PV_data["hour"]/12*(ma.pi)) # 時系列データをcosに変換
time_data = pd.concat([sin_data, cos_data], axis=1) # sin、cosデータを連結
name = ["sin", "cos"] # 列名
time_data.columns = name # 列名付与
PV_data = pd.concat([PV_data, time_data], axis=1) # 元データにsin、cosデータを連結

##データの整理
factor = 3 # 入力データの要素数
train_days = 90 # 学習データの日数
test_days = 30 # テストデータの日数
pred_days = 30 # 予測データの日数
time_stamps = PV_data["hour"]
time_stamp = time_stamps[:48]
row = len(PV_data) #　PV_dataの行数
col = len(PV_data.columns)#　PV_dataの列数

train_data = PV_data[48*(train_days - 30):48*train_days] # テストデータ
long_test_data = PV_data[48*train_days:48*(train_days + test_days)] # 学習データ
test_target_data = long_test_data["price[yen/kW30m]"]
test_PV_data = (long_test_data["PVout[kW]"].reset_index(drop=True))/2 # テスト時の目標のデータ
test_weather_data = preprocessing.minmax_scale(long_test_data["weather"])
test_weather_data = pd.DataFrame(test_weather_data)
test_input_data = pd.concat([long_test_data["sin"].reset_index(drop=True),test_weather_data,test_PV_data],axis=1) # テスト時の入力データ

train_target_data = train_data["price[yen/kW30m]"]
train_PV_data = (train_data["PVout[kW]"].reset_index(drop=True))/2 # 学習時の目標のデータ
train_weather_data = preprocessing.minmax_scale(train_data["weather"])
train_weather_data = pd.DataFrame(train_weather_data)
train_input_data = pd.concat([train_data["sin"].reset_index(drop=True),train_weather_data,train_PV_data],axis=1) # 学習時の入力データ

test_input_data = (test_input_data.values) # 型変換
test_target_data = (test_target_data.values) # 型変換
train_input_data = (train_input_data.values)# 型変換
train_target_data = (train_target_data.values)# 型変換

test_input_data = test_input_data.reshape((30, 48, factor)) # 3次元に変換
test_target_data = test_target_data.reshape((30, 48, 1)) # 3次元に変換
train_input_data = train_input_data.reshape((30, 48, factor)) # 3次元に変換
train_target_data = train_target_data.reshape((30, 48, 1)) # 3次元に変換


## LSTM
#ハイパーパラメータ
input_dim = factor # 入力データの要素数
output_dim = 1 # 出力データ数
len_sequence = 48 # 時系列の長さ
batch_size = 128 # ミニパッチサイズ
num_of_training_epochs = 3 # 3000 # 学習エポック数
learning_rate = 0.0001 # 学習率

# モデルの構築
LSTM_hidden_units_1 = 3 # 256 # 隠れ層（第一層）のユニット数
LSTM_hidden_units_2 = 3 # 128 # 隠れ層（第二層）のユニット数
LSTM_hidden_units_3 = 3 # 64 # 隠れ層（第三層）のユニット数
LSTM_model = Sequential()
LSTM_model.add(LSTM(LSTM_hidden_units_1, input_shape=(len_sequence, input_dim), activation = "relu", return_sequences=True)) # 1層目
LSTM_model.add(LSTM(LSTM_hidden_units_2, input_shape=(len_sequence, input_dim), activation = "relu", return_sequences=True)) # 2層目
LSTM_model.add(LSTM(LSTM_hidden_units_3, input_shape=(len_sequence, input_dim), activation = "relu", return_sequences=True)) # 3層目
LSTM_model.add(Dense(output_dim))
LSTM_model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))

# 学習
LSTM_model.fit(train_input_data, train_target_data, batch_size=batch_size, epochs=num_of_training_epochs, validation_split=0.001, verbose=0)
LSTM_model.save("LSTM_model_price")

# 試験
score = [] # errorを格納する配列
target=test_target_data
load_model_name = "LSTM_model_price" # LSTM_model or NN_model # cloudy_data sunny_data dataset_2019-2021
LSTM_model = keras.models.load_model(load_model_name)
testPredict_LSTM = LSTM_model.predict(test_input_data)

# 予測のRMSEを一日ごとに計算
for x in range(0, 30):
    testRMSE = ma.sqrt(mean_squared_error(target[:,x], testPredict_LSTM[:,x]))
    score.append(testRMSE)

# Graph描写
plt.plot(score, "red", drawstyle="steps-post")
plt.legend(["RMSE"]) # 凡例
plt.xlabel("day") # 横軸
plt.ylabel("RMSE[yen]") # 縦軸
plt.show()

#グラフ出力
price_result = pd.DataFrame(columns=["predict", "target"])

Predict_LSTM = py.reshape(testPredict_LSTM, (1440, 1))
Predict_LSTM = pd.DataFrame(Predict_LSTM)

target = py.reshape(test_target_data, (1440, 1))
target = pd.DataFrame(target)

price_result[["predict"]] = Predict_LSTM
price_result[["target"]] = target

price_result.to_csv('predicd_price.csv')