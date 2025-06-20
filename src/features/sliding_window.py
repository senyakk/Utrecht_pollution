import numpy as np

# note: this function was inspired by
# Brownlee, J. (2018, November 13). How to Develop LSTM Models for Time Series
# Forecasting. Machine Learning Mastery. https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

def split_sequences_multi_output(sequences, target, n_steps:int, n_steps_out:int):
    '''
    Splits multivariate time series data into input-output sequences.

    This function creates sequences of input data (X) and corresponding target outputs for
    forecasting tasks where both target variables and features are predicted. It extracts 
    `n_steps` time steps as input and predicts the next `n_steps_out` time steps for both 
    the target variable and the feature sequences.
    '''
    X, y_target, y_features = list(), list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps   # end of this input window
        out_end_ix = end_ix + n_steps_out  # end of the target window (1 step ahead)
        if out_end_ix > len(sequences):
            break

        seq_x = sequences[i:end_ix, :]
        X.append(seq_x)

        if target is not None:
            seq_q_target = target[end_ix:out_end_ix, 0]
            seq_q_features = sequences[end_ix:out_end_ix, :]
            y_target.append(seq_q_target)
            y_features.append(seq_q_features)
    
    return np.array(X), np.array(y_target), np.array(y_features)