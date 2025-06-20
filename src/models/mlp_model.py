import tensorflow as tf
from ..models.model import Model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
    

class MLPmodel(Model):
    """
    Class to build MLP
    By defaullt it has two hidden layers with relu activation functions.
    Adam optimiser is used
    """
    def __init__(self, X_train, n_steps_out = 1,
                        lr = 0.0001, dropout_rate = 0.2,
                        num_units1 = 64, num_units2 = 32):
        super().__init__(X_train, n_steps_out, lr, dropout_rate, num_units1, num_units2)

    def _create_model(self, X_train, n_steps_out,
                            lr, dropout_rate,
                            num_units1, num_units2):
        n_steps = X_train.shape[1]
        n_features = X_train.shape[2]

        model = Sequential()
        model.add(Dense(num_units1, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_units2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=n_steps_out))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mae', metrics=['mae'])
        self._model = model