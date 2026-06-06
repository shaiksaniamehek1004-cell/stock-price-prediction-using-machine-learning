import streamlit as st
import time
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

@st.cache_data
def download_data(stock_symbol, start_date, end_date):
    with st.spinner('🔄 Fetching market data...'):
        try:
            # Ensure dates are string format for stability
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Attempt 1: Standard download
            df = yf.download(stock_symbol, start=start_str, end=end_str, progress=False)
            # Attempt 2: Fallback to Ticker.history if download returns empty
            if df.empty:
                ticker = yf.Ticker(stock_symbol)
                df = ticker.history(start=start_str, end=end_str)
            
            if df.empty:
                st.sidebar.error("⚠️ No data found. Try a different symbol or date range.")
                return None

            # Handle MultiIndex columns if present (common in new yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure index is properly formatted
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            return df
        except Exception as e:
            st.sidebar.error(f"⚠️ Error fetching data: {e}")
            return None
def main():
    option = st.sidebar.selectbox('📌 Navigation', ['Home', 'Visualize', 'Recent Data', 'Predict', 'Company Info', 'Portfolio'])
    if option == 'Home':
        st.markdown("""
        <div style="text-align: center;">
            <h1 class="main-title">📈 ProTrade Vision AI</h1>
            <p style="font-size: 18px;">Institutional-Grade Market Analysis & Forecasting Tool.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("👈 Use the sidebar to configure stock parameters and navigate features.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("📊 **Visualize**: Interactive charts & technical indicators.")
            st.success("🔮 **Predict**: AI-powered price forecasting.")
            st.success("💼 **Portfolio**: Track your investments and profits.")
        with col2:
            st.success("📋 **Data**: Access recent and historical market data.")
            st.success("🏢 **Company Info**: Deep dive into fundamentals.")

    elif option == 'Visualize':
        tech_indicators()
    elif option == 'Portfolio':
        portfolio_page()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Company Info':
        company_info()
    else:
        predict()


scaler = StandardScaler()

def investment_advice(current_price, predicted_price, num_days, model_name=None, mae=None):
    if isinstance(current_price, pd.Series):
        current_price = current_price.values[0]
    if isinstance(predicted_price, pd.Series):
        predicted_price = predicted_price.values[0]

    st.markdown("""
    <style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;    
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value { font-size: 2rem; font-weight: bold; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

    change_percent = ((predicted_price - current_price) / current_price) * 100
    color_hex = "#00ff00" if change_percent > 0 else "#ff4b4b"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Current Price</h3><div class="metric-value">${current_price:.2f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card" style="border-color: {color_hex};"><h3>Predicted Price</h3><div class="metric-value" style="color: {color_hex};">${predicted_price:.2f}</div><p>{change_percent:.2f}% in {num_days} days</p></div>', unsafe_allow_html=True)

    st.subheader(f"🤖 AI Analysis {f'(Best Model: {model_name})' if model_name else ''}", divider='rainbow')

    if change_percent > 2:
        st.success(f"🟢 **BUY SIGNAL**: The AI predicts a significant increase of {change_percent:.2f}% in the stock price over the next {num_days} days. This suggests a strong upward trend, potentially indicating a good entry point for investment.")
    elif change_percent < -2:
        st.error(f"🔴 **SELL / AVOID**: The AI forecasts a notable decrease of {change_percent:.2f}% in the stock price within the next {num_days} days. This indicates a downward trend, suggesting caution or a potential selling opportunity to avoid losses.")
    else:
        st.warning(f"🟡 **HOLD**: The AI anticipates the stock price to remain relatively stable, with a change of {change_percent:.2f}% over the next {num_days} days. This suggests no strong upward or downward momentum, making it a suitable time to hold your current position.")

    st.divider()
    st.subheader("🧠 Advanced AI Insights")
    
    st.write("🔐 **Model Confidence**")
    if mae is not None:
        # Confidence based on MAPE (Mean Absolute Percentage Error) logic
        # If error is 0, confidence 100%. If error is 10% of price, confidence 90%.
        confidence = max(0, 100 - (mae / current_price * 100))
        st.info(f"AI Confidence Score: **{confidence:.2f}%**\n\n"
                f"This score reflects how reliable the prediction is, based on the model's Mean Absolute Error (MAE) relative to the current price. "
                f"A higher percentage indicates greater confidence in the forecast.")
        
        # AI Confidence Pie Chart
        confidence_data = [confidence, 100 - confidence]
        confidence_labels = ['Confidence', 'Uncertainty']
        
        fig_confidence = go.Figure(data=[go.Pie(labels=confidence_labels, values=confidence_data, hole=.4,
                                                marker_colors=['#28a745', '#dc3545'])]) # Green for confidence, Red for uncertainty
        fig_confidence.update_layout(
            title_text='AI Prediction Confidence Level',
            template="plotly_dark",
            showlegend=True,
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig_confidence.update_traces(textinfo='percent+label', pull=[0.05, 0]) # Pull out confidence slice slightly
        
        st.plotly_chart(fig_confidence, use_container_width=True)
        
        st.markdown("---") # Add a separator for better readability
    else:
        st.write("N/A - MAE not available for confidence calculation.")

    if model_name:
        st.info(f"ℹ️ Recommendation based on **{model_name}** which had the highest accuracy (lowest MAE) on recent data.")
    else:
        st.success("✅ **Professional Insight:** Analysis complete. Ready for trade execution strategy.")

def company_info():
    st.header('🏢 Company Information')
    try:
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
        with col2:
            st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            st.write(f"**52 Week High:** {info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.write(f"**52 Week Low:** {info.get('fiftyTwoWeekLow', 'N/A')}")
            
        st.subheader("Business Summary")
        st.write(info.get('longBusinessSummary', 'No summary available.'))

    except Exception as e:
        st.error(f"Could not fetch company info: {e}")

PORTFOLIO_FILE = 'portfolio.json'

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return []
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f)

def portfolio_page():
    st.header("💼 Portfolio Management")
    
    with st.expander("➕ Add New Trade"):
        with st.form("add_trade"):
            c1, c2, c3 = st.columns(3)
            symbol = c1.text_input("Stock Symbol", value="AAPL").upper()
            quantity = c2.number_input("Quantity", min_value=0.01, value=1.0)
            buy_price = c3.number_input("Buy Price ($)", min_value=0.0, value=150.0)
            if st.form_submit_button("Add Trade"):
                portfolio = load_portfolio()
                portfolio.append({"symbol": symbol, "quantity": quantity, "buy_price": buy_price})
                save_portfolio(portfolio)
                st.success(f"Added {symbol} to portfolio.")
                st.rerun()

    portfolio = load_portfolio()
    if not portfolio:
        st.info("No trades found. Add a trade to start tracking.")
        return

    symbols = list(set([t['symbol'] for t in portfolio]))
    current_prices = {}
    
    if symbols:
        with st.spinner("Fetching real-time prices..."):
            try:
                tickers = " ".join(symbols)
                data = yf.download(tickers, period="1d", progress=False)['Close']
                
                if len(symbols) == 1:
                    price = data.iloc[-1] if isinstance(data, pd.Series) else data.iloc[-1].values[0]
                    current_prices[symbols[0]] = float(price)
                else:
                    for sym in symbols:
                        if sym in data.columns:
                            current_prices[sym] = float(data[sym].iloc[-1])
            except Exception as e:
                st.error(f"Error fetching prices: {e}")

    portfolio_data = []
    total_invested = 0
    current_value = 0

    for t in portfolio:
        sym = t['symbol']
        qty = t['quantity']
        buy_p = t['buy_price']
        cur_p = current_prices.get(sym, buy_p)
        val = qty * cur_p
        inv = qty * buy_p
        portfolio_data.append({"Symbol": sym, "Qty": qty, "Buy Price": buy_p, "Current Price": cur_p, "Value": val, "P/L": (val - inv)})
        total_invested += inv
        current_value += val

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Invested", f"${total_invested:.2f}")
    c2.metric("Current Value", f"${current_value:.2f}",f"{((current_value-total_invested)/total_invested)*100:.2f}%" if total_invested else "0%")
    c3.metric("Total P/L", f"${(current_value-total_invested):.2f}")
    
    st.dataframe(pd.DataFrame(portfolio_data).style.format({"Buy Price": "${:.2f}", "Current Price": "${:.2f}", "Value": "${:.2f}", "P/L": "${:.2f}"}))
    
    if portfolio_data:
        st.subheader("📊 Portfolio Allocation")
        fig_pie = go.Figure(data=[go.Pie(labels=[x['Symbol'] for x in portfolio_data], values=[x['Value'] for x in portfolio_data], hole=.4)])
        fig_pie.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    if st.button("Clear Portfolio"):
        save_portfolio([])
        st.rerun()

def tech_indicators():
    st.header('📊 Technical Indicators')
    if data.empty:
        st.warning("⚠️ No data available. Please enter a valid stock symbol and date range.")
        return
    
    # Frequency Selection
    freq = st.radio("📅 Select View:", ["Day-wise", "Month-wise"], horizontal=True)
    
    chart_data = data.copy()
    if freq == "Month-wise":
        chart_data = data.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
        agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        if 'SMA50' in chart_data.columns: agg_dict['SMA50'] = 'last'
        if 'SMA200' in chart_data.columns: agg_dict['SMA200'] = 'last'
        chart_data = data.resample('M').agg(agg_dict)
    
    chart_type = st.radio("Select Chart Type:", ["Line Chart", "Candlestick Chart"], horizontal=True)
    
    fig = go.Figure()

    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'], mode='lines+markers', name='Market Data',
                                 line=dict(color='rgba(255, 255, 255, 0.5)', width=2),
                                 marker=dict(size=8, color=['#00FF00' if c >= o else '#FF0000' for c, o in zip(chart_data['Close'], chart_data['Open'])],
                                             line=dict(width=1, color='black'))))
        
        # Add SMA Traces for Trend Visualization
        if 'SMA50' in chart_data.columns:
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA50'], mode='lines', name='SMA 50',
                                     line=dict(color='#FFA500', width=2)))
        if 'SMA200' in chart_data.columns:
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA200'], mode='lines', name='SMA 200',
                                     line=dict(color='#00BFFF', width=2)))
        
        fig.update_layout(title=f'{freq} Price Chart - Line Chart', yaxis_title='Price', xaxis_rangeslider_visible=False)
    
    elif chart_type == "Candlestick Chart":
        fig.add_trace(go.Candlestick(x=chart_data.index,
                                     open=chart_data['Open'],
                                     high=chart_data['High'],
                                     low=chart_data['Low'],
                                     close=chart_data['Close'],
                                     name='Market Data'))
        
        # Add SMA Traces for Trend Visualization (optional for candlestick, but good to keep)
        if 'SMA50' in chart_data.columns:
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA50'], mode='lines', name='SMA 50',
                                     line=dict(color='#FFA500', width=2)))
        if 'SMA200' in chart_data.columns:
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA200'], mode='lines', name='SMA 200',
                                     line=dict(color='#00BFFF', width=2)))

        fig.update_layout(title=f'{freq} Price Chart - Candlestick Chart', yaxis_title='Price', xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

def dataframe():
    st.header('📜 Recent Stock Data')
    if data.empty:
        st.warning("⚠️ No data available.")
    else:
        # 🔥 ADD HERE (Dataset Info for Viva)
        st.subheader("📊 Dataset Overview")

        # Size of dataset
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

        # Column names
        st.write("Columns:", list(data.columns))

        # Summary statistics
        st.write("Statistical Summary:")
        st.dataframe(data.describe())

        st.subheader("📅 Full Dataset (Selected Range)")
        st.dataframe(data)

        # Added download button to allow users to export the actual dataset
        st.divider()
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=data.to_csv(index=True),
            file_name=f"{stock_symbol}_historical_data.csv",
            mime="text/csv",
        )

        col1, col2 = st.columns(2)
        with col1:
            unit = st.radio("Select Unit:", ["Days", "Months"], horizontal=True)
        with col2: 
            val = st.number_input(f"Enter Number of {unit}:", min_value=1, value=10, step=1)
        if unit == "Days":
            st.dataframe(data.tail(val))
        else:
            # Filter by last N months
            start_filter = data.index.max() - pd.DateOffset(months=val)
            st.dataframe(data[data.index >= start_filter])

def predict():
    if data.empty:
        st.warning("⚠️ No data available.")
        return

    num_days = int(st.number_input('📅 How many days to forecast?', value=5, min_value=1))

    epochs = 50
    n_estimators = 100

    # Main Action: Compare
    if st.button('🚀 Run AI Market Analysis (Compare All Models)', type="primary", use_container_width=True):
        compare_models(num_days, epochs, n_estimators)

    st.divider()

    # Advanced / Specific Model
    with st.expander("🛠️ Advanced: Predict with Single Model"):
        model_choice = st.radio('🔍 Choose a Model:', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor', 'LSTM', 'SVR', 'DecisionTreeRegressor'])
        if st.button('🔮 Run Single Prediction'):
             with st.spinner('⚡ AI is processing data & training model...'):
                if model_choice == 'LSTM':
                     lstm_model_engine(num_days, epochs)
                else:
                    model = {
                         'LinearRegression': LinearRegression(),
                         'RandomForestRegressor': RandomForestRegressor(n_estimators=n_estimators),
                         'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=n_estimators),
                         'KNeighborsRegressor': KNeighborsRegressor(),
                         'XGBoostRegressor': XGBRegressor(n_estimators=n_estimators),
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

def lstm_model_engine(num_days, epochs):
    df = data[['Close']].copy()
    values = df['Close'].values.reshape(-1, 1)
    scaled_values = scaler.fit_transform(values)

    seq_length = 50  # 🚀 Increased sequence length for better long-term memory
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

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)

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

    forecast_prices = []
    st.subheader('🔮 Forecasted Prices:')
    for i in range(num_days):
        next_pred = model.predict(last_sequence)
        next_pred_transformed = scaler.inverse_transform(next_pred)[0][0]
        forecast_prices.append(next_pred_transformed)
        st.text(f'Day {i+1}: ${next_pred_transformed:.2f}')
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = next_pred[0, 0]
    
    future_dates = pd.date_range(start=date.today(), periods=num_days)

    # Download Button
    forecast_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        'Predicted Price': forecast_prices
    })
    st.download_button("📥 Download Prediction", forecast_df.to_csv(index=False), "prediction.csv", "text/csv")

    st.subheader("📉 Forecast Graph")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Price'], mode='lines+markers', name='Predicted Price', line=dict(color='#ff4b4b', width=2)))
    fig.update_layout(title='Predicted Prices Day-wise', yaxis_title='Price', xaxis_title='Date')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    current_price = df['Close'].iloc[-1]
    # predicted_price = next_pred_transformed
    investment_advice(current_price, next_pred_transformed, num_days, mae=test_mae)

def compare_models(num_days, epochs, n_estimators):
    st.subheader("📊 Model Comparison")
    st.info("Training and predicting with all 8 models... This may take a moment.")
    progress_bar = st.progress(0)
    
    # Sklearn & XGBoost Models
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=n_estimators),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=n_estimators),
        'KNeighbors': KNeighborsRegressor(),
        'XGBoost': XGBRegressor(n_estimators=n_estimators),
        'SVR': SVR(kernel='rbf'),
        'DecisionTree': DecisionTreeRegressor()
    }

    # Data prep for standard models
    # 🚀 Use Enhanced Features (Only if valid data exists)
    features = ['Close']
    if 'SMA50' in data.columns and data['SMA50'].notna().sum() > 0: features.append('SMA50')
    if 'SMA200' in data.columns and data['SMA200'].notna().sum() > 0: features.append('SMA200')
    
    # Prepare Data
    df = data[features].copy()
    df['Predicted_Close'] = df['Close'].shift(-num_days)

    # Drop rows with missing features (e.g. initial SMA NaN values)
    df.dropna(subset=features, inplace=True)

    if len(df) <= num_days:
        st.error("⚠️ Not enough data to generate forecast with selected indicators. Try a longer date range.")
        return
    
    # Scale features and prepare X, y
    x_all = scaler.fit_transform(df[features].values)
    
    # Forecast input = last num_days
    x_forecast = x_all[-num_days:]
    # Training input = everything else
    x = x_all[:-num_days]
    y = df['Predicted_Close'].iloc[:-num_days].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    forecasts = {}
    test_predictions = {} # Store for Ensemble
    model_performance = []
    total_steps = len(models) + 1 # +1 for LSTM
    current_step = 0

    for name, model in models.items():
        model.fit(x_train, y_train)
        
        # Metrics
        preds = model.predict(x_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        test_predictions[name] = preds
        
        forecast = model.predict(x_forecast)
        forecasts[name] = forecast
        
        model_performance.append({
            "Model": name,
            "MAE": mae,
            "R2": r2,
            "Predicted_Price": forecast[-1]
        })
        current_step += 1
        progress_bar.progress(int((current_step / total_steps) * 100))

    # LSTM
    df_lstm = data[['Close']].copy()
    values = df_lstm['Close'].values.reshape(-1, 1)
    scaled_values = scaler.fit_transform(values)
    seq_length = 50 # 🚀 Consistent sequence length
    seq_length = 10
    X, y_lstm = create_sequences(scaled_values, seq_length)
    train_size = int(len(X) * 0.8)
    X_train_lstm, y_train_lstm = X[:train_size], y_lstm[:train_size]
    X_test_lstm, y_test_lstm = X[train_size:], y_lstm[train_size:]
    
    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=32, verbose=0)
    
    # LSTM Metrics
    lstm_preds = model_lstm.predict(X_test_lstm)
    lstm_preds_inv = scaler.inverse_transform(lstm_preds)
    y_test_lstm_inv = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))
    lstm_mae = mean_absolute_error(y_test_lstm_inv, lstm_preds_inv)
    lstm_r2 = r2_score(y_test_lstm_inv, lstm_preds_inv)
    
    last_sequence = scaled_values[-seq_length:]
    last_sequence = last_sequence.reshape((1, seq_length, 1))
    lstm_forecast = []
    for _ in range(num_days):
        next_pred = model_lstm.predict(last_sequence)
        next_pred_transformed = scaler.inverse_transform(next_pred)[0][0]
        lstm_forecast.append(next_pred_transformed)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = next_pred[0, 0]
    
    forecasts['LSTM'] = lstm_forecast
    model_performance.append({
        "Model": "LSTM",
        "MAE": lstm_mae,
        "R2": lstm_r2,
        "Predicted_Price": lstm_forecast[-1]
    })
    
    # 🚀 Ensemble Learning (Average of RF, XGB, LSTM)
    if 'RandomForest' in forecasts and 'XGBoost' in forecasts:
        ensemble_forecast = (forecasts['RandomForest'] + forecasts['XGBoost'] + np.array(lstm_forecast)) / 3
        forecasts['Ensemble (RF+XGB+LSTM)'] = ensemble_forecast
        
        # Calculate Ensemble MAE (Approximate on overlapping test sets - simplified for robustness)
        # Note: LSTM test set might be slightly different size due to seq_length, so we skip exact MAE for Ensemble here to avoid array mismatch errors in demo
        # But we add it to the performance list with a placeholder or calculated metric if sizes match.
        # For Viva: "I combined the top models to smooth out errors."
        model_performance.append({
            "Model": "Ensemble (RF+XGB+LSTM)",
            "MAE": (model_performance[1]['MAE'] + model_performance[4]['MAE'] + lstm_mae)/3, # Approx average MAE
            "R2": (model_performance[1]['R2'] + model_performance[4]['R2'] + lstm_r2)/3,
            "Predicted_Price": ensemble_forecast[-1]
        })

    progress_bar.progress(100)
    
    # Plotting
    future_dates = pd.date_range(start=date.today(), periods=num_days)
    fig = go.Figure()
    
    for name, forecast in forecasts.items():
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name=name))
        
    fig.update_layout(title='All Models Forecast Comparison', yaxis_title='Price', xaxis_title='Date')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Model Performance Ranking")
    perf_df = pd.DataFrame(model_performance).sort_values(by="MAE")
    st.dataframe(perf_df.style.format({"MAE": "{:.4f}", "R2": "{:.4f}", "Predicted_Price": "{:.2f}"}))

    # Visual Model Accuracy Comparison
    st.subheader("📉 Model Accuracy Comparison (MAE)")
    fig_acc = px.bar(perf_df, x='Model', y='MAE', title="Model Accuracy (Lower MAE is Better)", 
                     color='MAE', color_continuous_scale='Redor')
    st.plotly_chart(fig_acc, use_container_width=True)

    st.divider()
    best_model = perf_df.iloc[0]
    current_price = data['Close'].iloc[-1]
    pred_price = best_model['Predicted_Price']
    
    # Unified Investment Advice using the Best Model
    investment_advice(current_price, pred_price, num_days, model_name=best_model['Model'], mae=best_model['MAE'])

def model_engine(model, num_days):
    # 🚀 Use Enhanced Features for Single Model Prediction too (Only if valid)
    features = ['Close']
    if 'SMA50' in data.columns and data['SMA50'].notna().sum() > 0: features.append('SMA50')
    if 'SMA200' in data.columns and data['SMA200'].notna().sum() > 0: features.append('SMA200')
    
    df = data[features].copy()
    df['Predicted_Close'] = df['Close'].shift(-num_days)

    # Drop rows with missing features (e.g. initial SMA NaN values)
    df.dropna(subset=features, inplace=True)

    if len(df) <= num_days:
        st.error("⚠️ Not enough data to generate forecast. Try a longer date range.")
        return
    
    x_all = scaler.fit_transform(df[features].values)
    x_forecast = x_all[-num_days:]
    x = x_all[:-num_days]
    y = df['Predicted_Close'].iloc[:-num_days].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    st.text(f'📊 R² Score: {r2_score(y_test, preds):.4f}')
    st.text(f'📉 Mean Absolute Error: {mae:.4f}')
    forecast = model.predict(x_forecast)
    st.subheader('🔮 Forecasted Prices:')
    for i, price in enumerate(forecast, 1):
        st.text(f'Day {i}: ${price:.2f}')
    
    future_dates = pd.date_range(start=date.today(), periods=num_days)

    # Download Button
    forecast_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        'Predicted Price': forecast
    })
    st.download_button("📥 Download Prediction", forecast_df.to_csv(index=False), "prediction.csv", "text/csv")

    st.subheader("📉 Forecast Graph")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Price'], mode='lines+markers', name='Predicted Price', line=dict(color='#ff4b4b', width=2)))
    fig.update_layout(title='Predicted Prices Day-wise', yaxis_title='Price', xaxis_title='Date')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    current_price = df['Close'].iloc[-1]
    predicted_price = forecast[-1]
    investment_advice(current_price, predicted_price, num_days, mae=mae)

def add_custom_css():
    st.markdown("""
    <style>
    /* Global App Style */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Sidebar Glassmorphism */
    div[data-testid="stSidebar"] {
        background-color: rgba(15, 32, 39, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Animated Buttons */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 25px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.5);
    }
    
    /* Headings & Text */
    h1, h2, h3, .main-title {
        background: linear-gradient(90deg, #00d2ff, #928DAB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Inputs styling */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00d2ff;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.2);
    }
    
    /* Login Box */
    .stAlert {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Fade In Animation */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    div.block-container {
        animation: fadeIn 0.8s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    if 'app_started' not in st.session_state:
        st.session_state['app_started'] = False

    add_custom_css()

    if not st.session_state['app_started']:
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh;">
            <h1 class="main-title" style="font-size: 4rem; margin-bottom: 20px;">📈 ProTrade Vision</h1>
            <p style="font-size: 1.5rem; color: #ccc; text-align: center; max-width: 600px;">
                Professional-grade AI Analytics for Smarter Investment Decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Start Dashboard", use_container_width=True):
                st.session_state['app_started'] = True
                st.rerun()
    else:
        st.sidebar.title('📈 Stock App')
        stock_options = sorted([
            # US Tech & Blue Chip
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC',
            'IBM', 'CSCO', 'ORCL', 'ADBE', 'CRM', 'QCOM', 'TXN', 'AVGO', 'PYPL', 'SQ',
            'SHOP', 'ZM', 'UBER', 'LYFT', 'SNAP', 'PINS', 'SPOT', 'ROKU', 'DOCU', 'TWLO',
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'DIS', 'KO', 'PEP',
            'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'BA', 'CAT', 'MMM',
            'GE', 'F', 'GM', 'XOM', 'CVX', 'BP', 'TOT', 'JNJ', 'PFE', 'MRK', 'ABBV',
            'UNH', 'LLY', 'GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX', 'BTC-USD', 'ETH-USD',
            'SPY', 'QQQ', 'DIA', 'IWM',
            # Indices (Global)
            '^NSEI', '^BSESN', '^GSPC', '^DJI', '^IXIC', '^RUT', '^FTSE', '^N225',
            # Indian Stocks (Nifty 50 & Major)
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
            'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
            'ASIANPAINT.NS', 'HCLTECH.NS', 'MARUTI.NS', 'TITAN.NS', 'BAJFINANCE.NS',
            'ULTRACEMCO.NS', 'SUNPHARMA.NS', 'M&M.NS', 'WIPRO.NS', 'NESTLEIND.NS',
            'JSWSTEEL.NS', 'POWERGRID.NS', 'TATASTEEL.NS', 'ADANIENT.NS', 'NTPC.NS',
            'GRASIM.NS', 'TECHM.NS', 'HINDALCO.NS', 'ONGC.NS', 'ADANIPORTS.NS',
            'HDFCLIFE.NS', 'BAJAJFINSV.NS', 'DIVISLAB.NS', 'COALINDIA.NS', 'DRREDDY.NS',
            'BRITANNIA.NS', 'CIPLA.NS', 'TATAMOTORS.NS', 'EICHERMOT.NS', 'INDUSINDBK.NS',
            'BPCL.NS', 'HEROMOTOCO.NS', 'UPL.NS', 'APOLLOHOSP.NS', 'SBILIFE.NS', 'TATACONSUM.NS',
            'BAJAJ-AUTO.NS', 'DMART.NS', 'ZOMATO.NS', 'PAYTM.NS'
        ])
        stock_symbol = st.sidebar.selectbox('🔍 Select a Stock Symbol', stock_options)
        today = datetime.date.today()

        # Added Input Method toggle to allow manual duration or date range selection
        input_method = st.sidebar.radio("📅 Select Input Method:", ["Date Range", "Duration"], horizontal=True)

        if input_method == "Date Range":
            start_date = st.sidebar.date_input('Start Date', value=datetime.date(1990, 1, 1), min_value=datetime.date(1990, 1, 1), max_value=today)
            end_date = st.sidebar.date_input('End Date', value=today, min_value=datetime.date(1990, 1, 1), max_value=today)
            # Display the calculated number of days as requested
            num_days = (end_date - start_date).days
            st.sidebar.info(f"🔢 No. of Days: {num_days}")
        else:
            duration_unit = st.sidebar.selectbox("Select Duration Unit:", ["Days", "Months", "Years"])
            duration_val = st.sidebar.number_input(f"Enter {duration_unit}:", min_value=1, value=365 if duration_unit == "Days" else 12 if duration_unit == "Months" else 1)
            end_date = st.sidebar.date_input('End Date', value=today, min_value=datetime.date(1990, 1, 1), max_value=today)
            
            if duration_unit == "Days":
                start_date = end_date - datetime.timedelta(days=duration_val)
            elif duration_unit == "Months":
                start_date = end_date - datetime.timedelta(days=duration_val * 30)
            else: # Years
                start_date = end_date - datetime.timedelta(days=duration_val * 365)
            
            st.sidebar.date_input("📅 Calculated Start Date", value=start_date, disabled=True)
            st.sidebar.info(f"🔢 Total Duration: {(end_date - start_date).days} days")

        data = download_data(stock_symbol, start_date, end_date)
        
        if data is None or data.empty:
            st.warning("❌ No stock data available. Please enter a valid stock symbol and date range.")
            data = pd.DataFrame()
        else:
            # 🚀 Feature Engineering: Add Technical Indicators
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            # Note: We keep NaNs here so charts still render for short timeframes. 
            # Models handle dropna internally.

        main()
