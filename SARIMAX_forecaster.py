import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


class StockPriceForecaster:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.results = None

    def check_stationarity(self, timeseries):
        """Perform Dickey-Fuller test to check for stationarity"""
        result = adfuller(timeseries.dropna(), autolag='AIC')  
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        return 'Stationary' if result[1] < 0.05 else 'Non-Stationary'

    def preprocess_data(self):
        """Preprocess data by removing duplicates, sorting by date, and differencing to make stationary"""
        
        # Remove duplicates by index and sort data by Date
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        self.data = self.data.sort_index()

        # Interpolate any missing values in the 'Closing_Price' column
        self.data['Closing_Price'].interpolate(method='linear', inplace=True)

        # Check if the series is stationary
        print("Checking stationarity of the 'Closing_Price' series...")
        stationarity = self.check_stationarity(self.data['Closing_Price'])
        
        if stationarity == 'Non-Stationary':
            # Differencing to make the data stationary
            self.data['Differenced_Closing_Price'] = self.data['Closing_Price'].diff().dropna()
            print("Data differenced to achieve stationarity.")
        else:
            self.data['Differenced_Closing_Price'] = self.data['Closing_Price']
        
        print(self.data.head())

    def train_sarima_model(self, p=1, d=1, q=1, P=1, D=1, Q=1, s=7):
        """Train SARIMAX model with specified parameters"""
        # Ensure the differenced data is used for SARIMA
        self.model = SARIMAX(self.data['Differenced_Closing_Price'], 
                             order=(p, d, q), 
                             seasonal_order=(P, D, Q, s))
        self.results = self.model.fit()

    def forecast(self, steps=7):
        """Forecast the next 'steps' days"""
        forecast = self.results.get_forecast(steps=steps)

        # If differenced, re-integrate predictions
        if 'Differenced_Closing_Price' in self.data.columns:
            forecast_mean = forecast.predicted_mean.cumsum() + self.data['Closing_Price'].iloc[-1]
            forecast_ci = forecast.conf_int().cumsum() + self.data['Closing_Price'].iloc[-1]
        else:
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()

        # Generate forecast dates
        forecast_dates = pd.date_range(start=self.data.index[-1], periods=steps + 1, inclusive='right')

        # Plotting the forecast
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Closing_Price'], label='Observed', color='blue')
        plt.plot(forecast_dates, forecast_mean, label='Forecast', color='green', linestyle='--')
        plt.fill_between(forecast_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

