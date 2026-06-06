import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Title and Sidebar Info
st.title('📈 Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')

@st.cache_resource
def download_data(stock_symbol, start_date, end_date):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if df.empty:
            st.sidebar.error("⚠️ No data found for the given stock symbol and date range. Try again.")
            return None  
        return df
    except Exception as e:
        st.sidebar.error(f"⚠️ Error fetching data: {e}")
        return None

def main():
    option = st.sidebar.selectbox('📌 Select an option', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

stock_symbol = st.sidebar.text_input('🔍 Enter a Stock Symbol', value='AAPL').upper()
today = datetime.date.today()
duration = st.sidebar.number_input('📅 Enter duration (in days)', value=365)
start_date = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=start_date)
end_date = st.sidebar.date_input('End Date', value=today)

if st.sidebar.button('🔄 Fetch Data'):
    if start_date >= end_date:
        st.sidebar.error('⚠️ Start date must be earlier than the end date.')
    else:
        data = download_data(stock_symbol, start_date, end_date)
else:
    data = download_data(stock_symbol, start_date, end_date)

if data is None or data.empty:
    st.warning("❌ No stock data available. Please enter a valid stock symbol and date range.")
    data = pd.DataFrame()

scaler = StandardScaler()

def tech_indicators():
    st.header('📊 Technical Indicators')
    if data.empty:
        st.warning("⚠️ No data available. Please enter a valid stock symbol and date range.")
        return
    st.line_chart(data['Close'])

def dataframe():
    st.header('📜 Recent Stock Data')
    if data.empty:
        st.warning("⚠️ No data available.")
    else:
        st.dataframe(data.tail(10))

def predict():
    if data.empty:
        st.warning("⚠️ No data available.")
        return

    model_choice = st.radio('🔍 Choose a Model:', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor', 'LSTM', 'SVR', 'DecisionTreeRegressor'])
    num_days = int(st.number_input('📅 How many days to forecast?', value=5, min_value=1))

    if st.button('🔮 Predict'):
        if model_choice == 'LSTM':
            lstm_model_engine(num_days)
        else:
            model = {
                'LinearRegression': LinearRegression(),
                'RandomForestRegressor': RandomForestRegressor(),
                'ExtraTreesRegressor': ExtraTreesRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'XGBoostRegressor': XGBRegressor(),
                'SVR': SVR(kernel='rbf'),
                'DecisionTreeRegressor': DecisionTreeRegressor()
            }[model_choice]
            model_engine(model, num_days)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def lstm_model_engine(num_days):
    df = data[['Close']].copy()
    values = df['Close'].values.reshape(-1, 1)
    scaled_values = scaler.fit_transform(values)

    seq_length = 10
    X, y = create_sequences(scaled_values, seq_length)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    st.text(f'📊 Training R² Score: {train_r2:.4f}')
    st.text(f'📊 Test R² Score: {test_r2:.4f}')
    st.text(f'📉 Training MAE: {train_mae:.4f}')
    st.text(f'📉 Test MAE: {test_mae:.4f}')

    last_sequence = scaled_values[-seq_length:]
    last_sequence = last_sequence.reshape((1, seq_length, 1))

    st.subheader('🔮 Forecasted Prices:')
    for i in range(num_days):
        next_pred = model.predict(last_sequence)
        next_pred_transformed = scaler.inverse_transform(next_pred)[0][0]
        st.text(f'Day {i+1}: ${next_pred_transformed:.2f}')
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = next_pred[0, 0]

def model_engine(model, num_days):
    df = data[['Close']].copy()
    df['Predicted_Close'] = df['Close'].shift(-num_days)
    x = df.drop(['Predicted_Close'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num_days:]
    x = x[:-num_days]
    y = df['Predicted_Close'].dropna().values

    if len(y) < num_days:
        st.error("⚠️ Not enough data for prediction. Try a smaller forecast period.")
        return

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'📊 R² Score: {r2_score(y_test, preds):.4f}')
    st.text(f'📉 Mean Absolute Error: {mean_absolute_error(y_test, preds):.4f}')
    forecast = model.predict(x_forecast)
    st.subheader('🔮 Forecasted Prices:')
    for i, price in enumerate(forecast, 1):
        st.text(f'Day {i}: ${price:.2f}')

if __name__ == '__main__':
    main()
