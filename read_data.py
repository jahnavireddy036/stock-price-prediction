import pandas as pd

class ReadStockData:
    def __init__(self, path: str) -> None:
        """
        Initializes the ReadStockData class with the file path and reads the stock data.
        
        Parameters:
            path (str): Path to the CSV file containing stock data.
        """
        self.df = None  # Initialize as None to handle the case when file is not loaded
        self.path = path
        self.read_stock_data()  # Read and filter the data immediately after initialization

    def read_stock_data(self):
        """Reads the CSV file, filters stock data to include only data after 2015, and stores it as a dataframe."""
        try:
            print(f"Attempting to read the file at {self.path}...")
            # Read CSV and set 'Date' column as index with correct date format
            self.df = pd.read_csv(self.path, index_col='Date', parse_dates=['Date'], dayfirst=True)
            
            # Filter the data to include only rows after January 1, 2015
            self.df = self.df[self.df.index > '2010-01-01']

            print(f"Data successfully read and filtered from {self.path}")
        except Exception as e:
            print(f"Error reading the file at {self.path}: {e}")
