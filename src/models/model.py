import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Model(ABC):
    """
    This class is a parent class for models built using Keras
    """
    def __init__(self, X_train, n_steps_out, lr, dropout_rate, num_units1, num_units2):
        self._model = None
        self._create_model(X_train, n_steps_out, lr, dropout_rate, num_units1, num_units2)
    
    @abstractmethod
    def _create_model(self, X_train, n_steps_out, lr, dropout_rate, num_units1, num_units2):
        pass

    def get_compiled_model(self):
        """
        Returns Keras compiled model
        """
        return self._model
    
    def get_summary(self) -> None:
        """
        Prints summary of the model's configurations
        """
        print(self._model.summary)

    def train(self, X_train, y_train, X_val, y_val, epochs = 50, batch_size = 32) -> None:
        """
        Trains the model on the provided training and validation data.
        """
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        self._history = self._model.fit(X_train, y_train, epochs=epochs, callbacks=[earlyStopping],
                                        batch_size=batch_size, verbose=1,
                                        validation_data=(X_val, y_val), shuffle=False)
    
    def get_history(self):
        return self._history

    def predict(self, X_test, y_test, n_steps_predict, window_start, verbose=1, visual=0):
        """
        Predicts test data fro a given window for n steps.
        """
        if n_steps_predict is None:
            if visual:
                raise ValueError("Specify n_steps_predict for a visualization.")
            predictions = self._model.predict(X_test)
            if verbose:
                self._display_metrics(y_test, predictions)
            return None

        X_input = X_test[window_start:window_start+n_steps_predict]
        pred = self._model.predict(X_input)
        true_labels = y_test[window_start:window_start+n_steps_predict]
        
        if verbose:
            mse, mae = self._display_metrics(true_labels, pred)
        if visual:
            self._display_pred(true_labels, pred, n_steps_predict)
        
        return pred.flatten(), true_labels.flatten(), mse, mae

    def _display_metrics(self, y_test, predictions):
        """
        Calculated MSE and MAE metrics for model's predictions given true values.
        """
        test_mse = mean_squared_error(y_test, predictions)
        test_mae = mean_absolute_error(y_test, predictions)
        print(f'Test MSE: {test_mse}')
        print(f'Test MAE: {test_mae}')
        return test_mse, test_mae

    def _display_pred(self, y_test, predictions, n_steps_predict):
        """
        Plots the true and model's predicted values for n number of steps
        """
        plt.figure(figsize=(10, 6))
        time_future = np.arange(0, n_steps_predict)
        plt.plot(time_future, y_test.flatten(), label="true", color="green")
        plt.plot(time_future, predictions.flatten(), label="predicted", color="red", linestyle="--")

        plt.title("True and Predicted Values")
        plt.xlabel("Time (hours)")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    def plot_history(self) -> None:
        """
        Plots the history of the model training for inspecting training and validation losses
        """
        loss = self._history.history["loss"]
        val_loss = self._history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()