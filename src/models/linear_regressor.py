from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1_l2
from ..models.model import Model

class LinearRegressor(Model):
    """
    Class to build Elastic Net linear regressor
    It has one hidde layer with linear activation function.
    Adam optimiser is used
    """
    def __init__(self, X_train, n_steps_out = 1, lr = 0.001, dropout_rate = None, num_units1 = 1, num_units2 = None, l1_penalty=0.01, l2_penalty=0.01):
        self.l1 = l1_penalty
        self.l2 = l2_penalty
        super().__init__(X_train, n_steps_out, lr, dropout_rate, num_units1, num_units2)

    def _create_model(self, X_train, n_steps_out, lr, dropout_rate, num_units1, num_units2):

        n_steps = X_train.shape[1]
        n_features = X_train.shape[2]

        self._model = Sequential()
        self._model.add(Dense(num_units1, input_shape=(n_steps, n_features), 
                              activation='linear', kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2)))

        self._model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def predict(self, X_test, y_test, n_steps_predict, window_start, verbose=1, visual=0):
        if n_steps_predict is None:
            if visual:
                raise ValueError("Specify n_steps_predict for a visualization.")
            predictions = self._model.predict(X_test)
            if verbose:
                self._display_metrics(y_test, predictions)
            return None

        X_input = X_test[window_start:window_start+n_steps_predict]
        pred = self._model.predict(X_input)
        pred = pred[:, 2, :]
        true_labels = y_test[window_start:window_start+n_steps_predict]
        
        if verbose:
            mse, mae = self._display_metrics(true_labels, pred)
        if visual:
            self._display_pred(true_labels, pred, n_steps_predict)
        
        return pred.flatten(), true_labels.flatten(), mse, mae