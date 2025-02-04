import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class LSTMStockPriceForecaster:
    def __init__(self, data, look_back=60):       
        self.data = data
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.train_X, self.train_y, self.test_X, self.test_y = None, None, None, None

    def preprocess_data(self):   
        self.data.sort_index(inplace=True)
        self.data['Closing_Price'].interpolate(method='linear', inplace=True)       
        
        close_prices = self.data['Closing_Price'].values.reshape(-1, 1)
        self.scaler.fit(close_prices)
        scaled_data = self.scaler.transform(close_prices)
        
        # Prepare the data sequences for LSTM (input: look_back days, output: next day)
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i - self.look_back:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        # Splitting the data into training and test sets (80% train, 20% test)
        split_index = int(len(X) * 0.8)
        self.train_X, self.train_y = X[:split_index], y[:split_index]
        self.test_X, self.test_y = X[split_index:], y[split_index:]

        # Reshape the input data for LSTM [samples, time_steps, features]
        self.train_X = np.reshape(self.train_X, (self.train_X.shape[0], self.train_X.shape[1], 1))
        self.test_X = np.reshape(self.test_X, (self.test_X.shape[0], self.test_X.shape[1], 1))

    def build_lstm_model(self, units=50, epochs=10, batch_size=32):       
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(self.look_back, 1)))
        model.add(LSTM(units=units))
        model.add(Dense(1))
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        self.model = model

        # Train the model
        self.model.fit(self.train_X, self.train_y, epochs=epochs, batch_size=batch_size)

    def forecast(self, steps=30):       
        future_forecast = []
        current_batch = self.test_X[-1].reshape(1, self.look_back, 1)

        for _ in range(steps):
            future_prediction = self.model.predict(current_batch)[0]
            future_forecast.append(future_prediction[0])
            current_batch = np.append(current_batch[:, 1:, :], [[future_prediction]], axis=1)

        # Inverse transform the forecasted prices to original scale
        future_forecast = self.scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))
        return future_forecast

    def plot_results(self, future_steps=30):      
        # Inverse transform the test data and predictions to original scale
        true_test_prices = self.scaler.inverse_transform(self.test_y.reshape(-1, 1))
        test_predictions = self.model.predict(self.test_X)
        test_predictions = self.scaler.inverse_transform(test_predictions)

        # Plot the original data
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Closing_Price'], label='Original Data', color='blue')

        # Plot the test predictions (for dates corresponding to the test set)
        test_dates = self.data.index[-len(true_test_prices):]
        plt.plot(test_dates, test_predictions, label='Predicted Prices', color='orange')

        # Plot the future forecast
        future_forecast = self.forecast(steps=future_steps)
        future_dates = pd.date_range(start=self.data.index[-1], periods=future_steps + 1, inclusive='right')
        plt.plot(future_dates[:len(future_forecast)], future_forecast, label='Future Forecast', color='green', linestyle='--')

        # Labels, titles, and formatting
        plt.title('Stock Price Prediction using LSTM')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()



