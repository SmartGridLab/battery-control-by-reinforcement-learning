# Import standard library modules
from cgi import print_arguments, print_directory, test
import pandas as pd
import numpy as np
import math as ma
import datetime
import random
import os
import csv
import warnings
# Third-party library modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import keras
import tensorflow as tf
import properscoring as prscore
# Locally developed modules
from parameters import Parameters as p
from result_inputdata_reference import ResultInputDataReference as RIRD

class PV_Predict:
    def __init__(self):
        warnings.simplefilter('ignore')
        print("\n---PV出力予測プログラム開始---\n")
        # current_data.csvの日付データを読み込む
        date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
        latest_date = date_info['date'].max()
        self.latest_year = latest_date.year
        self.latest_month = latest_date.month
        self.latest_day = latest_date.day
        # Specify the target
        self.target = ["PVout"]
        # Soecify the predictors
        self.features = ['temperature', 'total precipitation', 'u-component of wind', 'v-component of wind', 'radiation flux', 'pressure', 'relative humidity', 
                        'yearSin', 'yearCos','monthSin', 'monthCos', 'hourSin', 'hourCos']

        # 現在の日付を取得
        self.RIRD = RIRD()
        self.year, self.month, self.day = self.RIRD.get_current_date()
        self.df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")


    # Set the randam variables
    def seed_everything(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)
        
    # Evaluate predictors and get gain for each features
    # - Data_energies contain the data for traing and test
    # - June in the data is for test which we put aside
    def get_feature_gain(self):
        target = ["PVout"]
        features = ['temperature', 'total precipitation', 'u-component of wind', 'v-component of wind', 'radiation flux', 'pressure', 'relative humidity', 
                    'yearSin', 'yearCos','monthSin', 'monthCos', 'hourSin', 'hourCos']
        df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
        train = df
        train_x,train_y = train[features],train[target]
        model = RandomForestRegressor(n_estimators=10, random_state=71) # these numbers are not optimized
        model.fit(train_x, train_y) # learning regression model
        feature_gain = model.feature_importances_ # get the gain for each predictors
        feature_gain = pd.DataFrame(feature_gain,index=features,columns=["gain"]).sort_values(ascending=False,by="gain") # sort the features in an ascending order
        feature_gain.to_csv('Battery-Control-By-Reinforcement-Learning/featureGainResult.csv')
        return feature_gain

    # Split into training and test data
    def make_df_w(self, mode):
        #　mode別で読み込むデータを変更
        if mode == "bid":
            df_w_all = pd.read_csv("Battery-Control-By-Reinforcement-Learning/weather_data_bid.csv")
        elif mode == "realtime":
            df_w_all = pd.read_csv("Battery-Control-By-Reinforcement-Learning/weather_data_realtime.csv")

        # 該当する日付部分を抽出
        df_w = df_w_all[(df_w_all['year'] == self.latest_year) &
                        (df_w_all['month'] == self.latest_month) &
                        (df_w_all['day'] == self.latest_day)]
        # インデックスをリセット
        df_w = df_w.reset_index(drop=True)
        #時系列のsin, cosを追加
        yearSin = np.sin(df_w["year"]/8760*(ma.pi))
        yearCos = np.cos(df_w["year"]/8760*(ma.pi))
        monthSin = np.sin(df_w["month"]/6*(ma.pi))
        monthCos = np.cos(df_w["month"]/6*(ma.pi))
        hourSin = np.sin(df_w["hour"]/12*(ma.pi))
        hourCos = np.cos(df_w["hour"]/12*(ma.pi))
        PVout = np.zeros((len(df_w),1)) # (48,1)

        time_data = pd.concat([yearSin, yearCos, monthSin, monthCos, hourSin, hourCos, pd.DataFrame(PVout)], axis=1)
        name = ['yearSin', 'yearCos','monthSin', 'monthCos', 'hourSin', 'hourCos', 'PVout'] # 列名
        time_data.columns = name # 列名付与
        #元のデータに統合
        df_w = pd.concat([df_w, time_data], axis=1)
        return df_w

    def train_test(self, use_col, df_w):
        # Load data from given csv file
        df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")
        
        # Split the data into Traing and Test
        train,test = df, df_w

        # Prepare space to store the data
        # For training
        # train = pd.concat([train,test[test.day<day]],axis=0)
        # For test
        # test = test[test.day==day]
        train_x, test_x = train[use_col], test[use_col]
        train_y = np.hstack([train[self.target], train[self.target]])
        test_y = np.hstack([test[self.target], test[self.target]])

        scaler = StandardScaler() # define normalization
        scaler.fit(train_x) # create model for normalization
        train_x = scaler.transform(train_x) # perform normalization using train_x
        test_x = scaler.transform(test_x) # perform normalization using test_x

        return train_x,test_x,train_y,test_y

    # get the loss function of neural network in LUBE
    def qd_objective(self, y_true, y_pred, lambda_, n_):
        y_true = y_true[:,0]
        y_u = y_pred[:,0]
        y_l = y_pred[:,1]
        
        K_HU = tf.maximum(0.,tf.sign(y_u - y_true))
        K_HL = tf.maximum(0.,tf.sign(y_true - y_l))
        K_H = tf.multiply(K_HU, K_HL)
        
        K_SU = tf.sigmoid(p.SOFTEN_ * (y_u - y_true))
        K_SL = tf.sigmoid(p.SOFTEN_ * (y_true - y_l))
        K_S = tf.multiply(K_SU, K_SL)
        
        MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/tf.reduce_sum(K_H)
        PICP_S = tf.reduce_mean(K_S)
        
        Loss_S = MPIW_c + lambda_ * n_ / (p.ALPHA_*(1-p.ALPHA_)) * (tf.pow(tf.maximum(0.,(1 - p.ALPHA_) - PICP_S), 2))
        return Loss_S

    # get neural network model to be used for LUBE
    def get_LUBE(self, number_of_features,LR,BETA, lambda_, n_):
        model = Sequential()
        model.add(Dense(100, input_dim=number_of_features, activation='relu',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))
        model.add(Dense(2, activation='linear',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3), 
                        bias_initializer=tf.keras.initializers.Constant(value=[3.,-3.]))) 
        opt = Adam(lr=LR, beta_1=BETA)
        model.compile(loss=lambda y_true, y_pred: self.qd_objective(y_true, y_pred, lambda_, n_), optimizer=opt)
        return model

    def mode_dependent_pv_predict(self, mode):
        self.seed_everything(p.SEED)
        # Get gain for each predictors
        feature_gain = self.get_feature_gain()

        # Prepare spaces to be stored the results
        result = pd.DataFrame(columns=["number_of_features","day","Loss","PICP","MPIW"])
        pred = pd.DataFrame(columns=["number_of_features","day","upper","lower"])
        # pv_predict = pd.DataFrame(columns=["year","month","day","hour","hourSin","hourCos","upper","lower","PVout","radiation flux","temperature",
        #                                   "total precipitation", "u-component of wind", "v-component of wind", "pressure", "relative humidity"])

        # Prepare time-series dat
        # df_wを取得
        df_w = self.make_df_w(mode)
        # 実行環境
        time = df_w[["hour"]]
        time_sin = np.sin(time*2*np.pi/24)
        time_cos = np.cos(time*2*np.pi/24)

        for i in range(p.N_VERIFICATION):
            for days in range(1,p.DAYS):
                print(days, i)

                # Prepare the space to store the predicted value
                pred_ = pd.DataFrame(columns=["number_of_features","day","upper","lower"])
                # Extract predictors from data set
                use_col = feature_gain.index[:p.NUMBER_OF_FEATURES]
                # Get traing and test data
                X_train,X_test,y_train,y_test = self.train_test(use_col, df_w)
                # coefficients n and λ
                n_= y_train.shape[0]
                lambda_ = p.N_LAMBDA*1/n_ 
                # output prediction intervals using LUBE
                model = self.get_LUBE(p.NUMBER_OF_FEATURES,p.LR,p.BETA, lambda_, n_)
                # history = model.fit(X_train, y_train, epochs=p.EPOCHS, batch_size=n_, verbose=0,  validation_split=0.)
                y_pred = model.predict(X_test)
                
                # organize forecast results
                pred_[["upper","lower"]] = y_pred
                pred_[["number_of_features","day"]] = p.NUMBER_OF_FEATURES,days
                pred_["verification"] = i
                pred = pd.concat([pred,pred_],axis=0)
                pv_predict_ = pd.DataFrame(columns=["year","month","day","hour","hourSin","hourCos","upper","lower","PVout","radiation flux","temperature",
                                                    "total precipitation", "u-component of wind", "v-component of wind", "pressure", "relative humidity"])

                if i == (p.N_VERIFICATION-1):
                    pv_predict_[["upper","lower"]] = y_pred
                    pv_predict_[["year","month","day"]] = df_w[["year","month","day"]]
                    pv_predict_[["hour"]] = time
                    pv_predict_[["hourSin"]] = time_sin
                    pv_predict_[["hourCos"]] = time_cos
                    pv_predict_[["radiation flux"]] = df_w[["radiation flux"]]
                    pv_predict_[["temperature"]] = df_w[["temperature"]]
                    pv_predict_[["total precipitation", "u-component of wind", "v-component of wind", "pressure", "relative humidity"]] = df_w[["Total precipitation", "u-component of wind", "v-component of wind", "Pressure", "Relative humidity"]]

                    #modify upper and lower
                    pv_predict_.loc[pv_predict_['lower'] < 0, 'lower'] = 0
                    pv_predict_.loc[pv_predict_['upper'] < 0, 'upper'] = 0
                    pv_predict_.loc[pv_predict_['hour'] < 4, 'upper'] = 0
                    pv_predict_.loc[pv_predict_['hour'] < 4, 'lower'] = 0
                    pv_predict_.loc[pv_predict_['hour'] > 19.5, 'upper'] = 0
                    pv_predict_.loc[pv_predict_['hour'] > 19.5, 'lower'] = 0

                    #lower, upper中央値算出
                    pv_predict_["PVout"] = (pv_predict_["upper"] + pv_predict_["lower"]) / 2
                    #--------------------------テストデータの作成--------------------------#
                    # if mode == "realtime":
                    #     pv_predict_.loc[(pv_predict_['hour'] >= 0.0) & (pv_predict_['hour'] <= 7.5), 'PVout'] = 0.0
                    #     pv_predict_.loc[(pv_predict_['hour'] >= 8.0) & (pv_predict_['hour'] <= 15.5), 'PVout'] = 1.5
                    #     pv_predict_.loc[(pv_predict_['hour'] >= 16.0) & (pv_predict_['hour'] <= 23.5), 'PVout'] = 0.0
                    # elif mode == "bid":
                    #     pv_predict_["PVout"] = 0
                    #     pv_predict_.loc[(pv_predict_['hour'] >= 0.0) & (pv_predict_['hour'] <= 7.5), 'PVout'] = 1.5
                    #     pv_predict_.loc[(pv_predict_['hour'] >= 8.0) & (pv_predict_['hour'] <= 15.5), 'PVout'] = 0.0
                    #     pv_predict_.loc[(pv_predict_['hour'] >= 16.0) & (pv_predict_['hour'] <= 23.5), 'PVout'] = 1.5
                    #--------------------------test--------------------------#
                    #print(pv_predict_["PVout"])

                    #delete dummy data
                    #pv_predict_.pop('dummy1')
                    #pv_predict = pd.concat([pv_predict,pv_predict_],axis=0)
            
            print(str(i+1)+"/"+str(p.N_VERIFICATION)+"完了")
        ## -------------------------- 人工的なテストデータの作成  -------------------------- ##
        # テストデータを作成
        # if mode == "bid":
        #     pv_predict_.loc[pv_predict_.index[:48], 'PVout'] = self.RIRD.PV_actual + 1.5
        # elif mode == "realtime":
        #     pv_predict_.loc[pv_predict_.index[:48], 'PVout'] = self.RIRD.PV_actual # realtimeは実測値に一致させる
        ## -------------------------- 人工的なテストデータの作成  -------------------------- ##
        pv_predict_.to_csv('Battery-Control-By-Reinforcement-Learning/pv_predict.csv', index=False)
        print("\n---PV出力予測プログラム終了---\n")