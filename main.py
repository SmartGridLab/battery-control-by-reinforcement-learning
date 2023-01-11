# Import standard library modules
from cgi import print_arguments, print_directory, test
import pandas as pd
import numpy as np
import math
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
import keras
import tensorflow as tf
import properscoring as prscore

# Locally developed modules
import parameters as p

# ignore warinings
warnings.simplefilter('ignore')

# Set the randam variables
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    
# Evaluate predictors and get gain for each features
# - Data_energies contain the data for traing and test
# - June in the data is for test which we put aside
def get_feature_gain():
    target = ["generation"]
    features = ['Humidity', 'WindSpeed', 'Temp', 'CloudCover', 'Rain', 'SolarIrradiation', 
                'yearSin', 'yearCos', 'daySin', 'dayCos','monthSin', 'monthCos', 'hourSin', 'hourCos']
    df = pd.read_csv("Pseudo_SampleData.csv")
    train = df[df.month!=6] # extract traning data (June is for test)
    train_x,train_y = train[features],train[target]
    model = RandomForestRegressor(n_estimators=10, random_state=71) # these numbers are not optimized
    model.fit(train_x, train_y) # learning regression model
    feature_gain = model.feature_importances_ # get the gain for each predictors
    feature_gain = pd.DataFrame(feature_gain,index=features,columns=["gain"]).sort_values(ascending=False,by="gain") # sort the features in an ascending order
    feature_gain.to_csv('featureGainResult.csv')
    return feature_gain

# Split into training and test data
def train_test(day,use_col):
    # Load data from given csv file
    df = pd.read_csv("Pseudo_SampleData.csv")
    
    # Split the data into Traing and Test
    #  - Utilize June for the test. Other months are for training 
    train,test = df[df.month!=6],df[df.month==6] 
    
    # Prepare space to store the data
    # For training
    train = pd.concat([train,test[test.day<day]],axis=0)
    # For test
    test= test[test.day==day]

    train_x,test_x=train[use_col],test[use_col]
    train_y = np.hstack([train[target],train[target]])
    test_y = np.hstack([test[target],test[target]])

    scaler = StandardScaler() # define normalization
    scaler.fit(train_x) # create model for normalization
    train_x = scaler.transform(train_x) # perform normalization using train_x
    test_x = scaler.transform(test_x) # perform normalization using test_x

    return train_x,test_x,train_y,test_y

# obtain MPIW, PICP, and Loss, which are evaluation indices for the prediction intervals
def qd_test(y_true, y_pred):
    y_true = y_true[:,0]
    y_u = y_pred[:,0]
    y_l = y_pred[:,1]

    #print(y_true)
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_true))
    K_HL = tf.maximum(0.,tf.sign(y_true - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    
    K_SU = tf.sigmoid(p.SOFTEN_ * (y_u - y_true))
    K_SL = tf.sigmoid(p.SOFTEN_ * (y_true - y_l))
    K_S = tf.multiply(K_SU, K_SL)
    
    MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/tf.reduce_sum(K_H)
    PICP_S = tf.reduce_mean(K_S)    
    Loss_S = MPIW_c + lambda_ * n_ / (p.ALPHA_*(1-p.ALPHA_)) * ((tf.maximum(0.,(1-p.ALPHA_) - PICP_S))**2)
    
    return Loss_S.numpy(),PICP_S.numpy(),MPIW_c.numpy()

# get the loss function of neural network in LUBE
def qd_objective(y_true, y_pred):
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
    
    Loss_S = MPIW_c + lambda_ * n_ / (p.ALPHA_*(1-p.ALPHA_)) * ((tf.maximum(0.,(1-p.ALPHA_) - PICP_S))**2)
    
    return Loss_S

# get neural network model to be used for LUBE
def get_LUBE(number_of_features,LR,BETA):
    model = Sequential()
    model.add(Dense(100, input_dim=number_of_features, activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))
    model.add(Dense(2, activation='linear',
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3), 
                    bias_initializer=tf.keras.initializers.Constant(value=[3.,-3.]))) 
    opt = tf.keras.optimizers.Adam(lr=LR, beta_1=BETA)
    model.compile(loss=qd_objective, optimizer=opt)
    return model

# prediction intervals generation by quantile regression
def quantile_regression_result(NUMBER_OF_FEATURES, SEED_QR, N_VERIFICATION,
                                         LOWER_ALPHA, UPPER_ALPHA, LR, M_TR, M_LE, N_E):
    
    # SEED : seed of random numbers
    # n_verification : the number of verification; to examine the variability of the modeling error
    # number_of_features : number of features used for quantile_regression modeling
    
    ## Parameters of quantile regression　
    # LOWER_ALPHA,UPPER_ALPHA　:　the alpha-quantile of the huber loss function and the quantile loss function　
    #                             at the upper and lower bounds of the prediction interval
    # lr : leraning rate
    # M_TR : maximum depth of the individual regression estimators in quantile_regression
    # M_LE : the minimum number of samples required to be at a leaf node in quantile_regression
    # N_E : the number of boosting stages to perform
    # For more information, please see the official https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    
    # load data and split into train and test data
    df = pd.read_csv("Pseudo_SampleData.csv")
    train,test = df[df.month!=6],df[(df.month==6)&(df.day<15)]
    # Soecify the features and target
    all_features = ['SolarIrradiation', 'hourSin', 'hourCos', 'yearCos', 'CloudCover',
       'yearSin', 'Humidity', 'Temp', 'monthCos', 'WindSpeed', 'daySin',
       'dayCos', 'monthSin', 'Rain']
    target = ["generation"]  
    
    # Prepare spaces to be stored the results
    result = pd.DataFrame()
    pred = pd.DataFrame()
    for i in range(p.N_VERIFICATION):
        # select features to use by number_of_features
        features = all_features[:NUMBER_OF_FEATURES]
        X_train,y_train = train[features],train[target]
        X_test,y_test = test[features],test[target]
        # model of the lower bound of the prediction intervals
        lower_model = GradientBoostingRegressor(loss="quantile",                   
                                                alpha=LOWER_ALPHA,
                                                random_state=SEED_QR+100*i,
                                                learning_rate=LR,
                                                max_depth=M_TR,
                                                min_samples_leaf=M_LE, 
                                                n_estimators=N_E,
                                                verbose=0)
        # model of the upper bound of the prediction intervals
        upper_model = GradientBoostingRegressor(loss="quantile",
                                                alpha=UPPER_ALPHA,
                                                random_state=SEED_QR+100*i,
                                                learning_rate=LR,
                                                max_depth=M_TR,
                                                min_samples_leaf=M_LE, 
                                                n_estimators=N_E,
                                                verbose=0)
        
        # train upper and lower bound models.
        lower_model.fit(X_train, y_train)
        upper_model.fit(X_train, y_train)
        # Record actual values on test set
        predictions = pd.DataFrame(y_test)
        # Predict
        predictions['lower'] = lower_model.predict(X_test)
        predictions['upper'] = upper_model.predict(X_test)
        predictions["PICP"] = 0
        predictions.loc[(predictions.upper>=predictions.generation) & (predictions.generation>=predictions.lower),"PICP"] = 1
        predictions["MPIW"] = predictions["upper"] - predictions["lower"]
        predictions['day'] = test["day"]
        predictions["mean"] = (predictions["lower"]+predictions["upper"])/2
        predictions["std"] = ((predictions["lower"]+predictions["upper"])/2 - predictions["lower"])/1.96
        crps = []
        for i in range(predictions.shape[0]):
            d = predictions.iloc[i]
            CRPSlist = prscore.crps_gaussian(d["generation"], mu=d["mean"], sig=d["std"])
            CRPS = CRPSlist.mean()
            crps.append(CRPS)
        predictions["crps"] = crps
        predictions["features"] = NUMBER_OF_FEATURES
        pred = pred.append(predictions)

        CRPS,MPIW,PICP,Day = [],[],[],[]
        for day,group in predictions.groupby("day"):
            MPIWmean = group[group.PICP>0].MPIW.mean()
            PICPmean = group.PICP.mean()
            CRPSmean = group.crps.mean()
            MPIW.append(MPIWmean)
            PICP.append(PICPmean)
            CRPS.append(CRPSmean)
            Day.append(day)
        Loss = MPIW +  5/(LOWER_ALPHA*UPPER_ALPHA)*np.where(np.array([0]*14)>np.array([UPPER_ALPHA]*14) -np.array(PICP),np.array([0]*14) ,np.array([UPPER_ALPHA]*14) -np.array(PICP))

        scores = pd.DataFrame(list(zip(Day,MPIW,PICP,CRPS)),columns=["day","MPIW","PICP","CRPS"])
        scores["features"] = NUMBER_OF_FEATURES
        result = result.append(scores)
    return pred,result

# Specify the target
target = ["generation"]
# Soecify the predictors
features = ['Humidity', 'WindSpeed', 'Temp', 'CloudCover', 'Rain', 'SolarIrradiation', 
            'yearSin', 'yearCos', 'daySin', 'dayCos','monthSin', 'monthCos', 'hourSin', 'hourCos']
# Generate randam value as stable one
seed_everything(p.SEED)
# Get gain for each predictors
feature_gain = get_feature_gain()

# Prepare spaces to be stored the results
result = pd.DataFrame(columns=["number_of_features","day","Loss","PICP","MPIW"])
pred = pd.DataFrame(columns=["number_of_features","day","upper","lower"])
testcsv = pd.DataFrame(columns=["year","month","day","hour","temp","rain","weather","PVout_true","price","sin","cos","Forecast","lower","upper","forecast_price","alpha","bata"])


# Prepare time-series data
date_base = datetime.date(2014, 6, 14)
time = np.arange(0, 24, 0.5).reshape((48, 1))
time_sin = np.sin(time*2*np.pi/24)
time_cos = np.cos(time*2*np.pi/24)

# 
for i in range(p.N_VERIFICATION):
    for day in range(1,p.DAYS):
        print(day, i)
        date_output = date_base + datetime.timedelta(days=day)

        # Prepare the space to store the predicted value
        pred_ = pd.DataFrame(columns=["number_of_features","day","upper","lower"])
        # Extract predictors from data set
        use_col = feature_gain.index[:p.NUMBER_OF_FEATURES]
        # Get traing and test data
        X_train,X_test,y_train,y_test = train_test(day,use_col)
        # coefficients n and λ
        n_= y_train.shape[0]
        lambda_ = p.N_LAMBDA*1/n_ 
        # output prediction intervals using LUBE
        model = get_LUBE(p.NUMBER_OF_FEATURES,p.LR,p.BETA)
        history = model.fit(X_train, y_train, epochs=p.EPOCHS, batch_size=n_, verbose=0,  validation_split=0.)
        y_pred = model.predict(X_test, verbose=0)
        Loss_S_,PICP_S_,MPIW_c_ = qd_test(y_test,y_pred)
        # organize forecast results
        pred_[["upper","lower"]] = y_pred
        pred_[["number_of_features","day"]] = p.NUMBER_OF_FEATURES,day
        pred_["verification"] = i
        pred = pd.concat([pred,pred_],axis=0)
        scores = pd.DataFrame([Loss_S_,PICP_S_,MPIW_c_],index=["Loss","PICP","MPIW"]).T
        scores[["number_of_features","day"]] = p.NUMBER_OF_FEATURES,day
        scores["verification"] = i
        result = pd.concat([result,scores],axis=0)
        
        #print(day, i)
        testcsv_ = pd.DataFrame(columns=["year","month","day","upper","PVout_true","lower","dummy1","hour"])
        if i == (p.N_VERIFICATION-1):
            testcsv_[["upper","lower"]] = y_pred
            testcsv_.loc[testcsv_['lower'] < 0, 'lower'] = 0
            testcsv_[["PVout_true","dummy1"]] = y_test
            testcsv_[["year","month","day"]] = date_output.year,date_output.month,date_output.day
            testcsv_[["hour"]] = time
            testcsv_[["sin"]] = time_sin
            testcsv_[["cos"]] = time_cos

            testcsv_.pop('dummy1')
            testcsv = pd.concat([testcsv,testcsv_],axis=0)

            

#testcsv_[["alpha", "bata"]] = 0

testcsv.to_csv('test.csv')
result.to_csv('LUBEresult.csv')

#testzone start
#handover = pd.DataFrame(columns=["year","month","day","hour","temp","rain","weather","PVout_true","price","sin","cos","Forecast","lower","upper","forecast_price","alpha","bata"])
#testzone end

pred,result = quantile_regression_result(p.NUMBER_OF_FEATURES, p.SEED_QR, p.N_VERIFICATION,
                                         p.LOWER_ALPHA, p.UPPER_ALPHA, p.LR, p.M_TR, p.M_LE, p.N_E)
result.to_csv('QRresult.csv')
