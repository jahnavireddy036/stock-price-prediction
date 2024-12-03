# train.py
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping
from model import create_lstm_model  # Import the model definition
from read_data import *  # Import data functions

# Function to save plots
def save_plot(figure, filename):
    """Helper function to save plot as an image file."""
    plt.savefig(filename)
    plt.close(figure)  # Close the figure to avoid memory issues

# Function to train the LSTM model for each stock ticker
def train_lstm_model_for_tickers(tickers, base_path, model, sequence_length=80, epochs=100, batch_size=64, patience=10):
    results = []
    os.makedirs('models', exist_ok=True)  # Directory to save models
    os.makedirs('plots', exist_ok=True)   # Directory to save plots

    for ticker in tickers:
        print(f"Processing {ticker}...")

        # Load and prepare the data
        df = read_stock_data(os.path.join(base_path, f"{ticker}.csv"))
        X, y, scaler = prepare_data_for_lstm(df, sequence_length)  # Data scaled

        # Split the data into training and test sets
        train_size = int(len(X) * 0.80)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # Configure early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Measure training time
        start_time = time.time()

        # Train the model with early stopping
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                            callbacks=[early_stopping], verbose=0)

        # Calculate training time
        training_time = time.time() - start_time

        # Evaluate the model on the test set
        predicted_test = model.predict(X_test)
        # Inverse scale the predictions and actual values
        predicted_test = scaler.inverse_transform(predicted_test)
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = np.mean((y_test_scaled - predicted_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_scaled - predicted_test))

        print(f"{ticker}: MSE={mse}, RMSE={rmse}, MAE={mae}")

        # Save the model and scaler
        model.save(f'models/{ticker}_lstm_model.h5')
        joblib.dump(scaler, f'models/{ticker}_scaler.pkl')

        # Store the results
        results.append({
            "ticker": ticker,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "TrainingTime": training_time
        })

        # Plot training and validation loss
        figure = plt.figure(figsize=(14, 7))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f"{ticker} - Train Loss vs Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        save_plot(figure, f'plots/{ticker}_train_loss_vs_val_loss.png')

        # Plot actual vs predicted prices for the test set
        figure = plt.figure(figsize=(14, 7))
        plt.plot(df.index[train_size + sequence_length:], y_test_scaled, label='Actual Price', color='blue')
        plt.plot(df.index[train_size + sequence_length:], predicted_test, label='Predicted Price', color='green')
        plt.title(f"{ticker} - Actual vs Predicted Prices")
        plt.legend()
        save_plot(figure, f'plots/{ticker}_actual_vs_predicted_prices.png')

    # Convert results to DataFrame and return
    results_df = pd.DataFrame(results)
    return results_df

# List of stock tickers to train the models
tickers = ['AAPL', 'GOOG', 'MSFT']  # Example tickers, adjust as needed
base_path = 'data'

# Create the LSTM model
model = create_lstm_model(sequence_length=80)

# Train the model for each ticker
results = train_lstm_model_for_tickers(tickers, base_path, model, sequence_length=80, epochs=100, batch_size=64)
print(results)
