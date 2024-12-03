# model.py
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def create_lstm_model(sequence_length=80):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
