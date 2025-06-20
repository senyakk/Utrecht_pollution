import tensorflow as tf
from ..models.model import Model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMmodel(Model):
    """
    Class to build LSTM
    By defaullt it has two hidden layers with a dropout layer for each. Adam optimiser is used.
    """
    def __init__(self, X_train, n_steps_out = 1,
                        lr = 0.00001, dropout_rate = 0.2,
                        num_units1 = 128, num_units2 = 64):
        super().__init__(X_train, n_steps_out, lr, dropout_rate, num_units1, num_units2)

    def _create_model(self, X_train, n_steps_out,
                        lr, dropout_rate,
                        num_units1, num_units2):
        n_steps = X_train.shape[1]
        n_features = X_train.shape[2]

        model = Sequential()
        model.add(LSTM(num_units1, return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(num_units2, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(n_steps_out))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mae', metrics=['mae'])
        self._model = model
