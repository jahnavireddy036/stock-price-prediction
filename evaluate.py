import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from read_data import *
from pre_process import *

class LSTMEvaluator:
    def __init__(self, tickers, base_path, sequence_length=80, plot_dir='plots'):
        """
        Initializes the evaluator class with parameters.

        Parameters:
            tickers (list): List of stock tickers to evaluate.
            base_path (str): Path to the directory where stock data files are located.
            sequence_length (int): Length of the sequence for LSTM input.
            plot_dir (str): Directory to save evaluation plots.
        """
        self.tickers = tickers
        self.base_path = base_path
        self.sequence_length = sequence_length
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)  # Create directory for plots

    def load_model_and_scaler(self, ticker):
        """
        Loads the trained model and scaler for a given ticker.

        Parameters:
            ticker (str): Stock ticker.

        Returns:
            tuple: Loaded model and scaler.
        """
        model = load_model(f'models/{ticker}_lstm_model.h5')
        scaler = joblib.load(f'models/{ticker}_scaler.pkl')
        return model, scaler

    def evaluate_ticker(self, ticker):
        """
        Evaluates the LSTM model for a single stock ticker.

        Parameters:
            ticker (str): Stock ticker for evaluation.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        print(f"Evaluating {ticker}...")

        # Load the model and scaler
        model, scaler = self.load_model_and_scaler(ticker)

        # Load and prepare the data
        data_obj = ReadStockData(os.path.join(self.base_path, f"{ticker}.csv"))
        df = data_obj.df 
        X, y, scaler = prepare_data_for_lstm(df, self.sequence_length)

        # Split the data into training and test sets
        train_size = int(len(X) * 0.80)
        X_test, y_test = X[train_size:], y[train_size:]

        # Evaluate the model on the test set
        predicted_test = model.predict(X_test)
        predicted_test = scaler.inverse_transform(predicted_test)
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = np.mean((y_test_scaled - predicted_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_scaled - predicted_test))

        print(f"{ticker}: MSE={mse}, RMSE={rmse}, MAE={mae}")

        # Save the results
        self.plot_actual_vs_predicted(df, ticker, y_test_scaled, predicted_test, train_size)
        self.plot_training_loss(ticker)

        return {"ticker": ticker, "MSE": mse, "RMSE": rmse, "MAE": mae}

    def plot_actual_vs_predicted(self, df, ticker, y_test_scaled, predicted_test, train_size):
        """
        Plots actual vs predicted prices for the test set.

        Parameters:
            df (pd.DataFrame): Stock data.
            ticker (str): Stock ticker.
            y_test_scaled (np.ndarray): Actual values of the test set (scaled back).
            predicted_test (np.ndarray): Predicted values of the test set.
            train_size (int): Size of the training data.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(df.index[train_size + self.sequence_length:], y_test_scaled, label='Actual Price', color='blue')
        plt.plot(df.index[train_size + self.sequence_length:], predicted_test, label='Predicted Price', color='green')
        plt.title(f"{ticker} - Actual vs Predicted Prices")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plot_path = f"{self.plot_dir}/{ticker}_actual_vs_predicted_prices.png"
        plt.savefig(plot_path)
        plt.close()  # Ensure the plot doesn't stay open in memory

    def plot_training_loss(self, ticker):
        """
        Plots training and validation loss if available.

        Parameters:
            ticker (str): Stock ticker.
        """
        history_path = f'models/{ticker}_training_history.pkl'
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
            loss_plot_path = f"{self.plot_dir}/{ticker}_train_loss_vs_val_loss.png"
            plt.savefig(loss_plot_path)
            plt.close()  # Ensure the plot doesn't stay open in memory

    def evaluate_all(self):
        """
        Evaluates the LSTM model for all stock tickers.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation metrics for all tickers.
        """
        results = []
        for ticker in self.tickers:
            results.append(self.evaluate_ticker(ticker))
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    tickers = ['AAPL', 'AMZN', 'IBM']  # Example tickers
    base_path = 'data'
    evaluator = LSTMEvaluator(tickers, base_path, sequence_length=80)

    # Evaluate models and print results
    results = evaluator.evaluate_all()
    print(results)
