# Stock Price Prediction Using Machine Learning

## ProTrade Vision AI – Intelligent Stock Market Analysis & Forecasting System

### Project Overview

ProTrade Vision AI is a machine learning-based stock market analysis and prediction system developed as a Final Year B.Tech Project.

The application collects real-time and historical stock market data from Yahoo Finance and applies multiple Machine Learning and Deep Learning algorithms to predict future stock prices. The system also provides technical analysis, portfolio management, company insights, market visualization, and AI-driven investment recommendations through an interactive Streamlit dashboard.

The objective of the project is to assist investors in making data-driven decisions by combining stock market analytics, machine learning prediction models, and financial visualization techniques.

---

## Problem Statement

Stock markets generate massive amounts of data every day, making it difficult for investors to accurately analyze trends and forecast future price movements.

Traditional analysis methods often fail to capture complex patterns in financial data.

This project addresses this challenge by using Machine Learning and Deep Learning models to analyze historical market behavior and predict future stock prices with improved accuracy.

---

## Objectives

* Collect historical stock market data automatically.
* Analyze stock market trends and technical indicators.
* Predict future stock prices using Machine Learning algorithms.
* Compare multiple prediction models.
* Identify the best-performing model using evaluation metrics.
* Provide AI-based investment recommendations.
* Develop an interactive dashboard for investors.

---

## Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

#### Data Processing

* Pandas
* NumPy

#### Machine Learning

* Scikit-Learn
* XGBoost

#### Deep Learning

* TensorFlow
* Keras (LSTM)

#### Data Visualization

* Plotly
* Streamlit

#### Data Collection

* Yahoo Finance API (yFinance)

---

## Machine Learning Models Implemented

### Linear Regression

Used as the baseline prediction model.

### Decision Tree Regressor

Captures non-linear relationships in stock market data.

### Random Forest Regressor

Uses ensemble learning to improve prediction stability and accuracy.

### Extra Trees Regressor

Reduces variance through randomized decision trees.

### K-Nearest Neighbors Regressor (KNN)

Predicts stock prices using similarity-based learning.

### Support Vector Regression (SVR)

Handles complex non-linear stock market patterns.

### XGBoost Regressor

Advanced gradient boosting algorithm for high-performance prediction.

### Long Short-Term Memory (LSTM)

Deep Learning model designed for time series forecasting and sequential financial data.

---

## Dataset Information

### Data Source

Yahoo Finance (yFinance)

### Features Used

* Open Price
* High Price
* Low Price
* Close Price
* Volume
* SMA50 (50-Day Moving Average)
* SMA200 (200-Day Moving Average)

### Target Variable

* Future Closing Price

---

## System Features

### Real-Time Market Data

* Automatic stock data collection from Yahoo Finance.
* Supports US stocks, Indian stocks, ETFs, indices, and cryptocurrencies.

### Technical Analysis

* Line Charts
* Candlestick Charts
* SMA50 Indicator
* SMA200 Indicator

### AI Price Prediction

* Multi-model prediction engine.
* Future stock price forecasting.
* Single-model and multi-model comparison modes.

### Model Comparison

The system compares:

* Linear Regression
* Random Forest
* Extra Trees
* Decision Tree
* SVR
* KNN
* XGBoost
* LSTM

and identifies the best-performing model.

### Portfolio Management

* Add stock holdings.
* Track profit and loss.
* Portfolio allocation visualization.
* Real-time valuation.

### Company Information

Displays:

* Sector
* Industry
* Market Capitalization
* P/E Ratio
* 52-Week High
* 52-Week Low
* Business Summary

### AI Investment Recommendation

Based on predicted price movement, the system generates:

* BUY Signal
* HOLD Signal
* SELL Signal

along with confidence analysis.

---

## Evaluation Metrics

### R² Score

Measures how well the model explains stock price variation.

Formula:

R² = 1 − (Residual Sum of Squares / Total Sum of Squares)

Higher value indicates better prediction performance.

---

### Mean Absolute Error (MAE)

Measures average prediction error.

Formula:

MAE = Average(|Actual − Predicted|)

Lower value indicates better accuracy.

---

## Methodology

### Step 1: Data Collection

Stock data is downloaded from Yahoo Finance using yFinance.

### Step 2: Data Preprocessing

* Handle missing values.
* Generate technical indicators.
* Normalize data using StandardScaler.

### Step 3: Feature Engineering

Create:

* SMA50
* SMA200

to improve model performance.

### Step 4: Model Training

Train multiple Machine Learning and Deep Learning models.

### Step 5: Model Evaluation

Evaluate models using:

* R² Score
* MAE

### Step 6: Prediction

Generate future stock price forecasts.

### Step 7: Recommendation

Provide AI-based investment suggestions.

---

## Project Architecture

User Input

↓

Yahoo Finance API

↓

Data Collection

↓

Data Preprocessing

↓

Feature Engineering

↓

Machine Learning Models

↓

Model Evaluation

↓

Prediction Engine

↓

Investment Recommendation

↓

Streamlit Dashboard

---

## Results

* Successfully implemented 8 prediction models.
* Developed an ensemble forecasting approach.
* Generated stock forecasts using historical market data.
* Compared model performances using MAE and R².
* Built a complete interactive dashboard for stock analysis.
* Provided automated investment recommendations.

---

## Key Learning Outcomes

* Machine Learning for Financial Analytics
* Time Series Forecasting
* Ensemble Learning
* Deep Learning using LSTM
* Feature Engineering
* Data Visualization
* Streamlit Application Development
* Model Evaluation Techniques
* Financial Data Analysis

---

## Future Enhancements

* Real-Time Prediction Using Live Market Data
* Sentiment Analysis Using Financial News
* Transformer-Based Forecasting Models
* Reinforcement Learning for Trading Strategies
* Cloud Deployment
* Mobile Application Integration

---

## Conclusion

ProTrade Vision AI successfully demonstrates the application of Machine Learning and Deep Learning techniques in stock market forecasting. By integrating multiple prediction models, technical indicators, portfolio management tools, and AI-powered investment recommendations, the system provides a comprehensive platform for financial analysis and decision support.

---

## Author

Sania Mehek

B.Tech – Artificial Intelligence & Data Science

Final Year Project

GitHub:
https://github.com/shaiksaniamehek1004-cell
