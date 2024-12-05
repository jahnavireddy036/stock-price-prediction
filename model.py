from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, sequence_length=80, units=100, dropout_rate=0.2, optimizer='nadam', loss='mean_squared_error'):
        """
        Initializes the LSTMModel class with given parameters.

        Parameters:
            sequence_length (int): The length of the input sequence.
            units (int): Number of units in the LSTM layers.
            dropout_rate (float): Dropout rate to prevent overfitting.
            optimizer (str): Optimizer used during model compilation.
            loss (str): Loss function used during model compilation.
        """
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.model = self._create_model()

    def _create_model(self):
        """Creates and compiles the LSTM model."""
        model = Sequential()

        # First LSTM Layer
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=(self.sequence_length, 1)))
        model.add(Dropout(self.dropout_rate))

        # Second LSTM Layer
        model.add(LSTM(units=self.units, return_sequences=False))
        model.add(Dropout(self.dropout_rate))

        # Dense Layer
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer=self.optimizer, loss=self.loss)

        return model

    def get_model(self):
        """Returns the compiled LSTM model."""
        return self.model
