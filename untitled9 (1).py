# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pMIkp4uh0nKOHYqj28a0GBuJwe2DkGZ2
"""

import pandas as pd
import numpy as np
df = pd.read_csv('WMT.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

float_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
df[float_columns] = df[float_columns].astype(float)
df['Volume'] = df['Volume'].astype(int)

print(df.isnull().sum())

df['MA20'] = df['Close'].rolling(window=20).mean()

df.sort_index(inplace=True)

print(df.head())

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

df = df.drop('MA20', axis=1)
data = df[['Adj Close', 'Volume']].values

df.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

sequence_length = 300  # You can adjust this
X, y = create_sequences(scaled_data, sequence_length)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 2)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=2
)

loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((len(predictions), 1))), axis=1))[:, 0]
actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), 1))), axis=1))[:, 0]

rmse = np.sqrt(np.mean((predictions - actual) ** 2))
print(f"Root Mean Squared Error: {rmse}")

max_error = np.max(np.abs(predictions - actual))
print(f"Maximum Error: {max_error}")

from sklearn.metrics import r2_score
r2 = r2_score(actual, predictions)
print(f"R-squared: {r2}")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()