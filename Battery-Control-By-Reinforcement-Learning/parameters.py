# For neural network
ALPHA_ = 0.05 #　measure of the confidence level of the prediction intervals
SOFTEN_ = 160 # parameter of Loss to smooth the NN learning curve
N_LAMBDA = 5 # parameter of Loss that ratios MPIW to PICP
BETA = 0.01 # the exponential decay rate for the 1st moment estimates in Adam optimization
EPOCHS = 100 # number of NN learning times
LR = 0.001 # learnning rate of NN
SEED = 2021 # seed of random numbers

# Simulation conditons
DAYS = 2 # days to be forecasted
N_VERIFICATION = 3 # the number of verification; to examine the variability of the modeling error

# For Quantile regression
LOWER_ALPHA = 0.025
UPPER_ALPHA = 0.975
LR=0.05
M_TR=3
M_LE=7
N_E=500
SEED_QR = 0

# Select the features from top. 
# - The number of features could be from 1 to 14. 
# - If you chose 1, only radiation is utilized for PV forcasting.
# - (1) 'SolarIrradiation', (2) 'hourSin', (3) 'hourCos', 'yearCos', 'CloudCover',....
#       'yearSin', 'Humidity', 'Temp', 'monthCos', 'WindSpeed', 'daySin',
#       'dayCos', 'monthSin', (14) 'Rain'
NUMBER_OF_FEATURES = 1
