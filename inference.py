import numpy as np
import joblib
import pandas as pd
from keras.models import load_model
import argparse

# Function to predict future stock prices
def predict_future_prices(ticker, model, scaler, sequence_length=80, n_days=10):
    # Load the data
    df = pd.read_csv(f"data/{ticker}.csv")
    columns_to_clean = ['Close/Last', 'Open', 'High', 'Low']
    
   
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].replace({'\$': ''}, regex=True).astype(float)
    df.set_index("Date", inplace=True)
    # Rename "Close/Last" to "Close"
    if 'Close/Last' in df.columns:
        df.rename(columns={'Close/Last': 'Close'}, inplace=True)
    # Extract the 'Close' prices
    close_prices = df['Close'].values

    # Normalize the data
    scaled_close = scaler.transform(close_prices.reshape(-1, 1))

    # Create the input sequence for the last n_days
    input_sequence = scaled_close[-sequence_length:]

    predictions = []
    for _ in range(n_days):
        # Reshape input sequence
        input_sequence = np.reshape(input_sequence, (1, sequence_length, 1))

        # Predict next day price
        predicted_price = model.predict(input_sequence)
        predictions.append(predicted_price[0, 0])

        # Update input sequence with the predicted price
        input_sequence = np.append(input_sequence[0][1:], predicted_price).reshape(-1, 1)

    # Inverse transform the predictions
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_prices

# Main function to handle command line arguments and inference
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict future stock prices")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g. 'AAPL')")
    parser.add_argument("n_days", type=int, help="Number of days to predict")
    args = parser.parse_args()

    ticker = args.ticker
    n_days = args.n_days

    # Load the model and scaler
    model = load_model(f'models/{ticker}_lstm_model.h5')
    scaler = joblib.load(f'models/{ticker}_scaler.pkl')

    # Predict future prices
    predicted_prices = predict_future_prices(ticker, model, scaler, sequence_length=80, n_days=n_days)

    # Print the predicted prices
    print(f"Predicted prices for the next {n_days} days for {ticker}:")
    for i, price in enumerate(predicted_prices):
        print(f"Day {i+1}: {price[0]:.2f}")

if __name__ == "__main__":
    main()
