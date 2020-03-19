                                # Part 1 - Data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('RELIANCE.NS.csv')
training_set = df.iloc[0:984,1:2].values
test_set = df.iloc[985: , 1:2].values


from sklearn.preprocessing import MinMaxScaler  #minmax is used for normalization
sc = MinMaxScaler(feature_range = (0,1))        # sc here means scaling
training_set_scale = sc.fit_transform(training_set)


X_train = []
y_train = []
for i in range(60, 983):
    X_train.append(training_set_scale[i-60:i, 0])
    y_train.append(training_set_scale[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


                                # Part 1 - LSTM

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)