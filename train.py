import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from model import LSTMModel 
from read_data import *  # 
from pre_process import * 

class StockLSTMTrainer:
    def __init__(self, tickers, base_path, sequence_length=80, epochs=100, batch_size=64, patience=10):
        """
        Initializes the trainer class with the parameters and prepares directories.

        Parameters:
            tickers (list): List of stock tickers to train models for.
            base_path (str): Path to the directory where stock data files are located.
            sequence_length (int): Length of the sequence for LSTM input.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            patience (int): Patience for early stopping.
        """
        self.tickers = tickers
        self.base_path = base_path
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model = LSTMModel(sequence_length=self.sequence_length).get_model()
        
        # Create directories for saving models and plots
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

    def save_plot(self, figure, filename):
        """Helper function to save plot as an image file."""
        plt.savefig(filename)
        plt.close(figure)  # Close the figure to avoid memory issues

    def train_model_for_ticker(self, ticker):
        """
        Trains the LSTM model for a single stock ticker.

        Parameters:
            ticker (str): Stock ticker for training the model.
        
        Returns:
            dict: Dictionary containing model evaluation results.
        """
        print(f"Processing {ticker}...")

        # Load and prepare the data
        data_obj = ReadStockData(os.path.join(self.base_path, f"{ticker}.csv"))
        df = data_obj.df 
        X, y, scaler = prepare_data_for_lstm(df, self.sequence_length)

        # Split the data into training and test sets
        train_size = int(len(X) * 0.80)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # Configure early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        # Measure training time
        start_time = time.time()

        # Train the model with early stopping
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2,
                                 callbacks=[early_stopping], verbose=0)

        # Calculate training time
        training_time = time.time() - start_time

        # Evaluate the model on the test set
        predicted_test = self.model.predict(X_test)
        # Inverse scale the predictions and actual values
        predicted_test = scaler.inverse_transform(predicted_test)
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = np.mean((y_test_scaled - predicted_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_scaled - predicted_test))

        print(f"{ticker}: MSE={mse}, RMSE={rmse}, MAE={mae}")

        # Save the model and scaler
        self.model.save(f'models/{ticker}_lstm_model.h5')
        joblib.dump(scaler, f'models/{ticker}_scaler.pkl')

        # Plot training and validation loss
        figure = plt.figure(figsize=(14, 7))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f"{ticker} - Train Loss vs Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        self.save_plot(figure, f'plots/{ticker}_train_loss_vs_val_loss.png')

        # Plot actual vs predicted prices for the test set
        figure = plt.figure(figsize=(14, 7))
        plt.plot(df.index[train_size + self.sequence_length:], y_test_scaled, label='Actual Price', color='blue')
        plt.plot(df.index[train_size + self.sequence_length:], predicted_test, label='Predicted Price', color='green')
        plt.title(f"{ticker} - Actual vs Predicted Prices")
        plt.legend()
        self.save_plot(figure, f'plots/{ticker}_actual_vs_predicted_prices.png')

        return {"ticker": ticker, "MSE": mse, "RMSE": rmse, "MAE": mae, "TrainingTime": training_time}

    def train_all_models(self):
        """
        Trains the LSTM model for all stock tickers.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results for all tickers.
        """
        results = []
        for ticker in self.tickers:
            results.append(self.train_model_for_ticker(ticker))

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        return results_df

# List of stock tickers to train the models
tickers = ['AAPL', 'AMZN', 'IBM']  # Example tickers, adjust as needed
base_path = 'data'

# Create an instance of the StockLSTMTrainer
trainer = StockLSTMTrainer(tickers, base_path, sequence_length=80, epochs=100, batch_size=64)

# Train the models and get the results
results = trainer.train_all_models()
print(results)
