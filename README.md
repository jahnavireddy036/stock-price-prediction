# Stock Price Prediction using LSTM

This project implements a machine learning pipeline to predict stock prices using Long Short-Term Memory (LSTM) models. The pipeline includes data preprocessing, model training, evaluation, prediction, and visualization.

## Project Structure

stock-price-prediction/
│
├── data/                         # Folder containing raw stock price data (CSV files).
│   └── AAPL.csv                  # Stock data for Apple (example).
│   └── GOOG.csv                  # Stock data for Google (example).
│   └── MSFT.csv                  # Stock data for Microsoft (example).
│   └── ...
│
├── models/                       # Folder to store trained LSTM models and scalers.
│   └── AAPL_lstm_model.h5        # LSTM model for Apple.
│   └── GOOG_lstm_model.h5        # LSTM model for Google.
│   └── MSFT_lstm_model.h5        # LSTM model for Microsoft.
│   └── AAPL_scaler.pkl           # Scaler used for Apple data.
│   └── GOOG_scaler.pkl           # Scaler used for Google data.
│   └── MSFT_scaler.pkl           # Scaler used for Microsoft data.
│
├── plots/                        # Folder to store generated plots.
│   └── AAPL_train_loss_vs_val_loss.png   # Train vs validation loss for Apple.
│   └── GOOG_train_loss_vs_val_loss.png   # Train vs validation loss for Google.
│   └── MSFT_train_loss_vs_val_loss.png   # Train vs validation loss for Microsoft.
│   └── AAPL_actual_vs_predicted_prices.png  # Actual vs predicted prices for Apple.
│   └── GOOG_actual_vs_predicted_prices.png  # Actual vs predicted prices for Google.
│   └── MSFT_actual_vs_predicted_prices.png  # Actual vs predicted prices for Microsoft.
│
├── eda.py                        # Script for exploratory data analysis (EDA) on stock data.
├── inference.py                  # Script to predict future stock prices using a trained model.
├── display_plots.py              # Script to display saved plots.
├── evaluate.py                   # Script to evaluate the performance of trained LSTM models.
├── model.py                      # Contains the architecture of the LSTM model.
├── read_data.py                  # Module to handle data loading and preprocessing for LSTM models.
├── train.py                      # Script to train the LSTM model for multiple stock tickers.
├── requirements.txt              # File containing Python package dependencies.
└── README.md                     # Project overview and instructions (this file).
