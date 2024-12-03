
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # Import the scaler

def clean_data(df):
    """
    Cleans the stock price data by:
    - Removing "$" symbol from specific columns if present.
    - Renaming the "Close/Last" column to "Close".
    """
    columns_to_clean = ['Close/Last', 'Open', 'High', 'Low']
    
    # Remove "$" symbol and convert to float
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].replace({'\$': ''}, regex=True).astype(float)
    
    # Rename "Close/Last" to "Close"
    if 'Close/Last' in df.columns:
        df.rename(columns={'Close/Last': 'Close'}, inplace=True)
    
    return df

def read_stock_data(file_path):
    """
    Reads and cleans stock data from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file containing stock data.
    
    Returns:
        pd.DataFrame: Cleaned stock data.
    """
    df = pd.read_csv(file_path)
    df.set_index("Date", inplace=True)
    df = clean_data(df)
    return df

def prepare_data_for_lstm(df, sequence_length=80):
    """
    Prepares data for LSTM model by creating sequences from the 'Close' price.
    Scales the data using MinMaxScaler.

    Parameters:
        df (pd.DataFrame): The cleaned stock data.
        sequence_length (int): Length of each sequence.
    
    Returns:
        np.ndarray: Processed input and output sequences for LSTM (scaled).
        MinMaxScaler: The fitted scaler for inverse transformation later.
    """
    close_prices = df['Close'].values.reshape(-1, 1)
    
    # Apply MinMax scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_prices)

    # Create sequences of data for LSTM
    X, y = [], []
    for i in range(sequence_length, len(scaled_close)):
        X.append(scaled_close[i-sequence_length:i, 0])  # Use scaled 'Close' prices directly
        y.append(scaled_close[i, 0])  # Scaled 'Close' price for prediction
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be 3D (samples, time steps, features) for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler
