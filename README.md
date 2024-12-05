# Stock Price Prediction using LSTM

This project implements a machine learning pipeline to predict stock prices using Long Short-Term Memory (LSTM) models. The pipeline includes data preprocessing, model training, evaluation, prediction, and visualization.

## Project Setup
1. Clone the repo
   ```bash
   git clone https://github.com/jahnavireddy036/stock-price-prediction.git
   ```
2. Install dependencies
   ```bash
   cd stock-price-prediction
   pip install -r requirements.txt
   ```
3. run cell in the notebook
   
## Folder & File Descriptions:

### **`data/`**
Contains the stock data files (`CSV` format) for different stock tickers. These data files are used to train and test the LSTM models. Each file should have the following columns: `Date`, `Open`, `Close`, `High`, `Low`, `Volume`.

### **`models/`**
Stores the trained LSTM models and the scalers (`.h5` and `.pkl` files). Each model corresponds to a stock ticker, and the scaler is used for normalizing the data.

### **`plots/`**
Contains the plots generated during the model training and evaluation process, including loss curves and actual vs predicted price plots.

## Python Scripts:

### **`eda.py`**
Used for exploring and visualizing the stock data. It provides basic statistics and charts to understand the data better.

### **`inference.py`**
Script for making predictions using the trained LSTM model. It accepts the ticker symbol and the number of days to predict as command-line parameters.

### **`display_plots.py`**
Displays saved plots from the `plots/` folder. It shows the loss curves and the actual vs predicted prices.

### **`evaluate.py`**
Evaluates the performance of trained LSTM models. It computes evaluation metrics like MSE, RMSE, and MAE, and generates plots of actual vs predicted stock prices.

### **`model.py`**
Defines the architecture of the LSTM model. This is where you define the layers, activation functions, and other configurations for the neural network.

### **`read_data.py`**
Contains functions for loading the stock data and preparing it for the LSTM model. This includes handling missing data, scaling, and generating sequences for training.

### **`train.py`**
This is the main script for training the LSTM models for multiple stock tickers. It loads the stock data, builds the LSTM model using `model.py`, trains the model, and saves the models and scalers to disk.

### **`requirements.txt`**
Specifies the Python dependencies required for this project. It includes libraries like `numpy`, `pandas`, `matplotlib`, `keras`, and `tensorflow`.


### **`README.md`**
This file. It provides an overview of the project, describes the folder structure, and gives instructions on how to use the scripts.

## Example Command Usage:

### **Training the Model**: 
To train the LSTM model for stock tickers, run the following command:
```bash
python train.py
```
### **Model Evaluation**:
To evaluate the trained LSTM models, run:
```bash
python evaluate.py
```

### **Making Predictions:**
To predict stock prices for the next n days, run:
```bash
python inference.py AAPL 10
```
Where AAPL is the stock ticker and 10 is the number of days to predict.

### **Displaying Plots:**
To display the saved plots, run
```bash
python display_plots.py
```

## Requirements

This project requires Python 3 and the following packages:

- `numpy`
- `pandas`
- `matplotlib`
- `keras`
- `tensorflow`
- `joblib`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt



