import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from read_data import read_stock_data, prepare_data_for_lstm 
import os

# Function to evaluate the model
def evaluate_lstm_model(tickers, base_path, sequence_length=80, plot_dir='plots'):
    results = []
    os.makedirs(plot_dir, exist_ok=True)  # Directory to save plots

    for ticker in tickers:
        print(f"Evaluating {ticker}...")

        # Load the model and scaler
        model = load_model(f'models/{ticker}_lstm_model.h5')
        scaler = joblib.load(f'models/{ticker}_scaler.pkl')

        # Load and prepare the data
        df = read_stock_data(f"{base_path}/{ticker}.csv")
        X, y, scaler = prepare_data_for_lstm(df, sequence_length)

        # Split the data into training and test sets
        train_size = int(len(X) * 0.80)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # Evaluate the model on the test set
        predicted_test = model.predict(X_test)
        predicted_test = scaler.inverse_transform(predicted_test)
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = np.mean((y_test_scaled - predicted_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_scaled - predicted_test))

        print(f"{ticker}: MSE={mse}, RMSE={rmse}, MAE={mae}")

        # Store the results
        results.append({
            "ticker": ticker,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae
        })

        # Plot actual vs predicted prices for the test set
        plt.figure(figsize=(14, 7))
        plt.plot(df.index[train_size + sequence_length:], y_test_scaled, label='Actual Price', color='blue')
        plt.plot(df.index[train_size + sequence_length:], predicted_test, label='Predicted Price', color='green')
        plt.title(f"{ticker} - Actual vs Predicted Prices")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plot_path = f"{plot_dir}/{ticker}_actual_vs_predicted_prices.png"
        plt.savefig(plot_path)
        plt.close()  # Close the plot to avoid memory issues in loops

        # Plot training and validation loss (if available)
        # If you saved the loss during training, load it here for visualization
        history_path = f'models/{ticker}_training_history.pkl'  # Assuming you saved it during training
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history = joblib.load(f)
            plt.figure(figsize=(14, 7))
            plt.plot(history['loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f"{ticker} - Train Loss vs Validation Loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            loss_plot_path = f"{plot_dir}/{ticker}_train_loss_vs_val_loss.png"
            plt.savefig(loss_plot_path)
            plt.close()

    # Convert results to DataFrame and return
    results_df = pd.DataFrame(results)
    return results_df

# List of stock tickers to evaluate the models
tickers = ['AAPL', 'GOOG', 'MSFT']  # Example tickers, adjust as needed
base_path = 'data'

# Call the evaluation function
results = evaluate_lstm_model(tickers, base_path, sequence_length=80)
print(results)
