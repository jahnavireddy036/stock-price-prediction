import os
import pandas as pd

def clean_data(df):
    """
    Cleans the stock price data by:
    - Removing "$" symbol from specific columns if present.
    - Renaming the "Close/Last" column to "Close".
    
    Parameters:
        df (pd.DataFrame): The dataframe to be cleaned.
    
    Returns:
        pd.DataFrame: The cleaned dataframe.
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

def read_data(data_folder_path):
    """
    Reads stock price CSV files from a folder and returns a dictionary with stock ticker as the key and dataframe as the value.
    
    Parameters:
        data_folder_path (str): Path to the folder containing the CSV files.
    
    Returns:
        dict: A dictionary with stock tickers as keys and their cleaned dataframes as values.
    """
    stock_data = {}
    
    for file_name in os.listdir(data_folder_path):
        if file_name.endswith('.csv'):
            # Extract stock ticker from file name
            stock_ticker = file_name.split('.')[0]
            
            # Read the CSV file
            file_path = os.path.join(data_folder_path, file_name)
            df = pd.read_csv(file_path)
            
            # Clean the data
            df = clean_data(df)
            
            # Add to dictionary
            stock_data[stock_ticker] = df
    
    return stock_data

