import streamlit as st
import pandas as pd
import time
from SARIMAX_forecaster import StockPriceForecaster
from LSTM_forecaster import LSTMStockPriceForecaster
import matplotlib.pyplot as plt

st.set_page_config(page_icon=':bar_chart:', layout='wide')
st.title(':bar_chart: Stock Market Movements')
st.markdown('<style> div.block-container{padding-top:2rem;text-align:center;} </style>', 
            unsafe_allow_html=True)

# Load dataset
data = pd.read_csv('joined_df.csv')
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  
data.set_index('Date', inplace=True)  
distinct_tickers = data['Ticker_Symbol'].drop_duplicates().tolist()

# Sidebar 
with st.sidebar:
    ticker = st.selectbox("Select Ticker", distinct_tickers)
    model = st.selectbox("Select a model to use", ['SARIMAX', 'LSTM'])
    days_to_forecast = st.slider('Number of days to Forecast', 1, 20)

# data filter
if model == 'SARIMAX':
    try:
        filtered_data = data[data['Ticker_Symbol'] == ticker]
        filtered_data['Closing_Price'] = pd.to_numeric(filtered_data['Closing_Price'], errors='coerce')
        filtered_data.dropna(subset=['Closing_Price'], inplace=True)

        # confirm if data exists after filtering
        if filtered_data.empty:
            st.error(f"No valid data available for : {ticker}")
        else:
            filtered_data = filtered_data[['Closing_Price']]            
            # StockPriceForecaster instance
            forecaster = StockPriceForecaster(filtered_data)
            forecaster.preprocess_data()
            st.write("Original Data Plot")
            st.line_chart(filtered_data['Closing_Price'])

            # Check for stationarity and decide whether to difference the data
            st.write("Differenced Data Plot (after preprocessing):")
            if 'Differenced_Closing_Price' in forecaster.data.columns:
                differenced_data = forecaster.data['Differenced_Closing_Price'].dropna()
                st.line_chart(differenced_data)

            # Train SARIMA model and forecast
            forecaster.train_sarima_model()         
            st.write(f"Forecasting the next {days_to_forecast} days for {ticker}")
            plt.figure(figsize=(10, 6))  
            forecaster.forecast(steps=days_to_forecast)          
            st.pyplot(plt)
    except:
        st.toast('OOPS! The Selected ticker {ticker}has no enough data to make predictions',icon='⚠️')
        time.sleep(4.9)
        st.toast('Kindly Select Another Ticker!',icon='😭')
        time.sleep(4.9)
        st.toast('Thank You!', icon='🙏')
        time.sleep(5.9)
        st.info(f'OOPS! {ticker} has no enough data to make predictions! Consider selecting another Ticker.')

else:
     try:
        filtered_data = data[data['Ticker_Symbol'] == ticker]
        filtered_data['Closing_Price'] = pd.to_numeric(filtered_data['Closing_Price'], errors='coerce')
        filtered_data.dropna(subset=['Closing_Price'], inplace=True)

        # LSTMStockPriceForecaster instance
        forecaster = LSTMStockPriceForecaster(filtered_data)
        forecaster.preprocess_data()
        forecaster.build_lstm_model(units=50, epochs=10, batch_size=32)
        # predictions 
        st.write(f"Forecasting the next {days_to_forecast} days for {ticker}")       
        future_steps = days_to_forecast
        forecaster.plot_results(future_steps=future_steps)        
        st.pyplot(plt.gcf())  # plt.gcf() gets the current active figure

     except Exception as e:
        st.info(f"Encountered an error: {e}")