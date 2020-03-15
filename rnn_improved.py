# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:56:09 2018

@author: souvik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

from keras.layers import Dense,LSTM
from keras.models import Sequential

regressor = Sequential()
regressor.add(LSTM(units = 50,dropout =0.2,recurrent_dropout=0.2,return_sequences=True, input_shape = (X_train.shape[1],1)))
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,dropout =0.2,recurrent_dropout=0.2,return_sequences = True))
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,dropout =0.2,recurrent_dropout=0.2, return_sequences = True))
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,dropout =0.2,recurrent_dropout=0.2))
#regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.iloc[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,:])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(test_set, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
    