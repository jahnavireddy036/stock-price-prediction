from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import plotly.graph_objs as go
import plotly.io as pio
import os

# Initialize Flask app and configurations
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tradeadmin'  
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Initialize paths and constants
base_path = "data/"
sequence_length = 80

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    ticker = request.form['ticker']
    n_days = int(request.form['days'])
    plot_html_full, plot_html_zoomed = predict_for_ticker(ticker, n_days)
    return render_template('result.html', plot_full=plot_html_full, plot_zoomed=plot_html_zoomed)

# Utility functions
def predict_for_ticker(ticker, n_days):
    model, scaler = load_model_and_scaler(ticker)
    df = load_and_process_data(ticker)
    
    close_prices = df['Close'].values
    scaled_close = scaler.transform(close_prices.reshape(-1, 1))

    # Create sequences for LSTM input
    X = create_sequences(scaled_close)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.80)
    X_train, X_test = X[:train_size], X[train_size:]

    # Predict prices
    predicted_test = model.predict(X_test)
    predicted_test = scaler.inverse_transform(predicted_test)

    # Plotting with Plotly
    actual_prices = scaler.inverse_transform(scaled_close)
    dates = df.index

    # Generate plots
    plot_html_full, plot_html_zoomed = generate_plots(dates, actual_prices, predicted_test, train_size, n_days)
    
    return plot_html_full, plot_html_zoomed

def load_model_and_scaler(ticker):
    """Load LSTM model and scaler for the given ticker."""
    model = load_model(f'models/{ticker}_lstm_model.h5')
    scaler = joblib.load(f'models/{ticker}_scaler.pkl')
    return model, scaler

def load_and_process_data(ticker):
    """Load and preprocess stock data."""
    df = pd.read_csv(f"data/{ticker}.csv", index_col='Date', parse_dates=['Date'], dayfirst=True)
    df = df[df.index > '2010-01-01']
    return df

def create_sequences(data):
    """Create sequences of data for LSTM input."""
    X = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
    return np.array(X)

def generate_plots(dates, actual_prices, predicted_prices, train_size, n_days):
    """Generate Plotly plots for actual vs predicted prices."""
    # Full plot
    full_end_index = train_size + sequence_length + n_days
    trace_actual = go.Scatter(
        x=dates[:full_end_index],
        y=actual_prices[:full_end_index].flatten(),
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    )
    trace_predicted = go.Scatter(
        x=dates[train_size + sequence_length:full_end_index],
        y=predicted_prices[:n_days].flatten(),
        mode='lines',
        name=f'Predicted Price for {n_days} Days',
        line=dict(color='green')
    )
    layout_full = go.Layout(
        title=f'Historical vs Predicted Prices',
        xaxis={'title': 'Date'},
        yaxis={'title': 'Price'},
    )
    fig_full = go.Figure(data=[trace_actual, trace_predicted], layout=layout_full)
    plot_html_full = pio.to_html(fig_full, full_html=False)

    # Zoomed plot for n-days prediction
    trace_zoomed_actual = go.Scatter(
        x=dates[train_size + sequence_length: full_end_index],
        y=actual_prices[train_size + sequence_length: full_end_index].flatten(),
        mode='lines',
        name='Actual Price (Test)',
        line=dict(color='blue')
    )
    trace_zoomed_predicted = go.Scatter(
        x=dates[train_size + sequence_length: full_end_index],
        y=predicted_prices[:n_days].flatten(),
        mode='lines',
        name=f'Predicted Price for {n_days} Days',
        line=dict(color='green')
    )
    layout_zoomed = go.Layout(
        title=f'{n_days} Days of Prediction vs Test',
        xaxis={'title': 'Date'},
        yaxis={'title': 'Price'},
    )
    fig_zoomed = go.Figure(data=[trace_zoomed_actual, trace_zoomed_predicted], layout=layout_zoomed)
    plot_html_zoomed = pio.to_html(fig_zoomed, full_html=False)

    return plot_html_full, plot_html_zoomed



# Run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
