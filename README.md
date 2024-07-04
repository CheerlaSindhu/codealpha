import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

# Define the ticker symbol
ticker = 'AAPL'

# Load the data
df = pdr.get_data_yahoo(ticker, start='2010-01-01', end='2023-01-01')
# Extract the 'Close' column for training
data = df.filter(['Close'])

# Convert the dataframe to a numpy array
dataset = data.values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Define the number of days to use for prediction
look_back = 60

# Create the scaled training data set
train_data = scaled_data[:int(len(dataset)*0.8), :]

# Split the data into x_train and y_train
x_train, y_train = [], []
for i in range(look_back, len(train_data)):
    x_train.append(train_data[i-look_back:i, 0])
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data (LSTM expects 3D input)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
model = Sequential()

# Add LSTM layer and Dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Fit the model
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Create the test data set
test_data = scaled_data[int(len(dataset)*0.8) - look_back:, :]

# Split the data into x_test and y_test
x_test = []
y_test = dataset[int(len(dataset)*0.8):, :]
for i in range(look_back, len(test_data)):
    x_test.append(test_data[i-look_back:i, 0])

# Convert x_test to a numpy array
x_test = np.array(x_test)

# Reshape the data (LSTM expects 3D input)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# Plot the data
train = data[:int(len(dataset)*0.8)]
valid = data[int(len(dataset)*0.8):]
valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
