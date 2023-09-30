import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense
import sklearn.metrics

SHEET_ID = '1CAUr5wAQRprvLhT8V8BH3syiFeKdtYKvK_QycPI-OVo'

def select_company(comp_name):
    # Reliance
    if comp_name.lower() == "reliance":
        SHEET_NAME = 'Reliance'

    # State Bank of India
    elif comp_name.lower() == "state bank of india":
        SHEET_NAME = 'SBI'

    # Yes Bank
    elif comp_name.lower() == "yes bank":
        SHEET_NAME = 'Yesbank'

    # Tata Motors
    elif comp_name.lower() == "tata motors":
        SHEET_NAME = 'TataMotors'

    # Zomato
    elif comp_name.lower() == "zomato":
        SHEET_NAME = 'Zomato'

    # Kotak Bank
    elif comp_name.lower() == "kotak bank":
        SHEET_NAME = 'KotakBank'

    # Bharat Heavy Electronics Limited
    elif comp_name.lower() == "bharat heavy electricals limited":
        SHEET_NAME = 'BHEL'

    # Reliance Power
    elif comp_name.lower() == "reliance power":
        SHEET_NAME = 'ReliancePower'

    # ICICI Bank
    elif comp_name.lower() == "icici bank":
        SHEET_NAME = 'ICICI'

    # National Thermal Power Corporation
    elif comp_name.lower() == "ntpc":
        SHEET_NAME = 'NTPC'

    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    df = pd.read_csv(url)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    return df

compname = input('Enter Company Name: ')

df = select_company(compname)
df_adjusted = df.loc[1:,]
df_adjusted[["Open", "High", "Low", "Close", "Volume"]] = df_adjusted[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric)
df_adjusted["Date"] = pd.to_datetime(df_adjusted["Date"])

df = df_adjusted

open_prices = df['Open']
values = open_prices.values
training_data_len = math.ceil(len(values)* 0.8)

#preprocessing using minmaxscaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))
train_data = scaled_data[0: training_data_len, :]

#getting the training set and testing set separated
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#prep the testing set
test_data = scaled_data[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size= 2, epochs= 15)

#evaluation
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#error metrics
mse = sklearn.metrics.mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)
mae = sklearn.metrics.mean_absolute_error(y_test, predictions)
r2 = sklearn.metrics.r2_score(y_test,predictions)

data= df.filter(['Open'])
train = data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Company - Model Prediciton Comparison')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Opening Price', fontsize=18)
plt.plot(train['Open'] , color='red')
plt.plot(valid['Open'] , color='yellow')
plt.plot(valid['Predictions'] , color='green')
plt.legend(['Train','Validation', 'Predictions'], loc='lower right')
plt.show()

valid.tail(15)

new_df = df.filter(['Open'])
last_60_days=new_df[-60:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test=[]
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
pred_price = model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)
print(f'Opening Price of {compname} tomorrow:{pred_price}')
