import numpy as np
import joblib
import pandas as pd
from keras.models import load_model
import argparse


class StockPricePredictor:
    def __init__(self, ticker, model_path, scaler_path, sequence_length=80):
        """
        Initializes the StockPricePredictor class.

        Parameters:
            ticker (str): Stock ticker symbol (e.g. 'AAPL').
            model_path (str): Path to the saved LSTM model.
            scaler_path (str): Path to the saved scaler.
            sequence_length (int): Length of the sequence for LSTM input.
        """
        self.ticker = ticker
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.sequence_length = sequence_length

        # Load the model and scaler
        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def preprocess_data(self):
        """
        Loads and preprocesses the stock data (removes symbols, cleans columns, etc.)

        Returns:
            np.ndarray: Scaled 'Close' prices.
        """
        # Load the data
      
        df = pd.read_csv(f"data/{self.ticker}.csv", index_col='Date', parse_dates=['Date'], dayfirst=True)
        df = df[df.index > '2010-01-01']
        # Extract and scale the 'Close' prices
        close_prices = df['Close'].values
        scaled_close = self.scaler.transform(close_prices.reshape(-1, 1))

        return scaled_close

    def predict_future_prices(self, n_days=10):
        """
        Predicts future stock prices for the next n_days.

        Parameters:
            n_days (int): Number of days to predict.

        Returns:
            np.ndarray: Predicted future stock prices.
        """
        scaled_close = self.preprocess_data()

        # Create the input sequence for the last 'sequence_length' days
        input_sequence = scaled_close[-self.sequence_length:]

        predictions = []
        for _ in range(n_days):
            # Reshape input sequence to match the model's expected input
            input_sequence = np.reshape(input_sequence, (1, self.sequence_length, 1))

            # Predict next day's price
            predicted_price = self.model.predict(input_sequence)
            predictions.append(predicted_price[0, 0])

            # Update the input sequence with the predicted price for the next iteration
            input_sequence = np.append(input_sequence[0][1:], predicted_price).reshape(-1, 1)

        # Inverse transform the predictions to get the original scale prices
        predicted_prices = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predicted_prices

    def display_predictions(self, predicted_prices):
        """
        Displays the predicted stock prices.

        Parameters:
            predicted_prices (np.ndarray): The predicted stock prices.
        """
        print(f"Predicted prices for the next {len(predicted_prices)} days for {self.ticker}:")
        for i, price in enumerate(predicted_prices):
            print(f"Day {i+1}: {price[0]:.2f}")

class Inference:
    def __init__(self):
        """Initializes the Inference class to handle command-line arguments."""
        self.parser = argparse.ArgumentParser(description="Predict future stock prices")
        self.parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g. 'AAPL')")
        self.parser.add_argument("n_days", type=int, help="Number of days to predict")
        self.args = self.parser.parse_args()

    def run_inference(self):
        """Run the stock price prediction using the provided arguments."""
        ticker = self.args.ticker
        n_days = self.args.n_days

        # Set paths for model and scaler
        model_path = f'models/{ticker}_lstm_model.h5'
        scaler_path = f'models/{ticker}_scaler.pkl'

        # Initialize the StockPricePredictor
        predictor = StockPricePredictor(ticker, model_path, scaler_path)

        # Predict future stock prices
        predicted_prices = predictor.predict_future_prices(n_days)

        # Display the predicted prices
        predictor.display_predictions(predicted_prices)

if __name__ == "__main__":
    inference = Inference()
    inference.run_inference()
