# python -m src.models.main

from ..features.tr_featureEng import FeatureEngineeringTraining
from ..models.mlp_model import MLPmodel
from ..models.lstm_model import LSTMmodel
from ..models.linear_regressor import LinearRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import csv
import os
from typing import List
# from codecarbon import EmissionsTracker


def prepare_data(pollutant:str, selected_features:List[str]):
    """
    Creates a FeatureEngineering object for training data.
    Returns the data which is ready for inputting into the model, fitted scalers on features and targets,
    and fitteed PCA.

    pollutant: 'O3' or 'NO2'
    selected features: list of features selected to train the model on
    """

    fe = FeatureEngineeringTraining('combined.csv')
    fe.extract_features(selected_features=selected_features, target=[pollutant],
                        lags=None, rolling_windows=[4,12,24], drop_ogs=0)
    # processed:
    # X_train_seq, y_train_seq_target, y_train_seq_features,
    # X_val_seq, y_val_seq_target, y_val_seq_features,
    # X_test_seq, y_test_seq_target, y_test_seq_features
    # (last only used if predicting features - for that, the models have to be modified)
    processed = fe.process(train_test=0.8, train_val=0.75, n_steps=3, n_steps_out=1)
    print(f"Feature engineering for {pollutant} finished.")

    return processed, fe.get_fitted_scaler(True), fe.get_fitted_scaler(False), fe.get_fitted_PCA()


def predict_pollutant(processed:list, n_steps_predict:int, months:List[int], model_name:str, epochs:int):
    """
    Testing pipeline for training seleceted model on the preprocessed data and returning predictions
    """
    
    X_train, y_train, _, X_val, y_val, _, X_test, y_test, _ = processed

    # tracker = EmissionsTracker() # keep track of energy consumption and CO2 emissions
    # tracker.start()

    if model_name == 'lr':
        model = LinearRegressor(X_train)
    elif model_name == 'mlp':
        model = MLPmodel(X_train)    # specify hyperparameters: n_steps_out, lr, dropout_rate, num_units1, num_units2
    elif model_name == 'lstm':
        model = LSTMmodel(X_train)
    else:
        print("Incorrect model's name.")

    if model is not None:
        model.get_summary()
        model.train(X_train, y_train, X_val, y_val, epochs=epochs)
        model_history = model.get_history()

        true_m = []
        pred_m = []

        for month in months:
            window_start = (month-1) * 30 * 24
            pred, true, _, _ = model.predict(X_test, y_test, n_steps_predict=n_steps_predict, window_start=window_start)  # specify time steps to predict
            true_m.append(true)
            pred_m.append(pred)
        
        # aggregate mae and mse over the months
        mse = mean_squared_error(np.concatenate(true_m), np.concatenate(pred_m))
        mae = mean_absolute_error(np.concatenate(true_m), np.concatenate(pred_m))
    
        print(f"{model_name} finished.")

    # tracker.stop()

    return true_m, [pred_m, mse, mae, model_history], model.get_compiled_model()

def prepare_dashboard_performance(lin_O3, mlp_O3, lstm_O3, lin_NO2, mlp_NO2, lstm_NO2):
    """
    Takes 6 models and saves their performance into a csv "history" files
    """
    models = ['Elastic Net', 'MLP', 'LSTM']

    if lin_O3 is None and lin_NO2 is None:
        lin_O3 = mlp_O3
        lin_NO2 = mlp_NO2

    mse_O3 = [lin_O3[1], mlp_O3[1], lstm_O3[1]]
    mae_O3 = [lin_O3[2], mlp_O3[2], lstm_O3[2]]
    mse_NO2 = [lin_NO2[1], mlp_NO2[1], lstm_NO2[1]]
    mae_NO2 = [lin_NO2[2], mlp_NO2[2], lstm_NO2[2]]

    # SAVE PERFORMANCE METRICS (MAE, MSE) FOR O3 AND NO2 MODELS
    rows = []
    for i in range(len(models)):
        row = [
            models[i],
            mse_O3[i],
            mae_O3[i],
            mse_NO2[i],
            mae_NO2[i]
        ]
        rows.append(row)
    
    cwd = os.getcwd()
    csv_file = os.path.join(cwd, 'results/performance.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model_name", "mse_loss_O3", "mae_loss_O3", "mse_loss_NO2", "mae_loss_NO2"])
        writer.writerows(rows)
    
    # SAVE MODEL TRAINING HISTORIES
    histories_O3 = [lin_O3[3], mlp_O3[3], lstm_O3[3]]
    histories_NO2 = [lin_NO2[3], mlp_NO2[3], lstm_NO2[3]]
    for i in range(len(models)):
        # save model training histories for O3 
        history_df_O3 = pd.DataFrame(histories_O3[i].history)
        history_df_O3.to_csv(os.path.join(cwd, f"results/history_{models[i]}_O3.csv"), index=False)

        # save model training histories for NO2
        history_df_NO2 = pd.DataFrame(histories_NO2[i].history)
        history_df_NO2.to_csv(os.path.join(cwd, f"results/history_{models[i]}_NO2.csv"), index=False)


def prepare_dashboard_predictions(n_steps_predict, true_O3_m, lin_O3, mlp_O3, lstm_O3, true_NO2_m, lin_NO2, mlp_NO2, lstm_NO2, o3_scaler, no2_scaler, months):
    """
    Takes number of prediction steps, true labels of pollutants, 6 models, 
    target features scalers and months for which predictions should be made. 
    
    Calculates the predictions for each model and saves them into a csv "results" files
    """
    # FOR PLOTTING PREDICTIONS OF ALL TYPES OF MODELS FOR BOTH O3 AND NO2
    models = ['Elastic Net'] * n_steps_predict + ['MLP'] * n_steps_predict + ['LSTM'] * n_steps_predict

    time_values = list(range(1, n_steps_predict+1)) * 3

    # SAVE PREDICTIONS FOR EACH MONTH IN A SEPARATE FILE
    for i in range(len(months)):
        month = months[i]

        true_O3 = true_O3_m[i]
        true_O3 = o3_scaler.inverse_transform(true_O3.reshape(-1, 1)).flatten()
        true_O3 = np.tile(true_O3, 3)
        pred_O3 = np.concatenate((o3_scaler.inverse_transform(lin_O3[0][i].reshape(-1, 1)).flatten(),
                                o3_scaler.inverse_transform(mlp_O3[0][i].reshape(-1, 1)).flatten(),
                                o3_scaler.inverse_transform(lstm_O3[0][i].reshape(-1, 1)).flatten()))

        true_NO2 = true_NO2_m[i]
        true_NO2 = o3_scaler.inverse_transform(true_NO2.reshape(-1, 1)).flatten()
        true_NO2 = np.tile(true_NO2, 3)
        pred_NO2 = np.concatenate((no2_scaler.inverse_transform(lin_NO2[0][i].reshape(-1, 1)).flatten(),
                                no2_scaler.inverse_transform(mlp_NO2[0][i].reshape(-1, 1)).flatten(),
                                no2_scaler.inverse_transform(lstm_NO2[0][i].reshape(-1, 1)).flatten()))

        rows = []
        for j in range(n_steps_predict*3):
            row = [
                models[j],
                time_values[j],
                true_O3[j],
                pred_O3[j],
                true_NO2[j],
                pred_NO2[j],
            ]
            rows.append(row)
        
        cwd = os.getcwd()
        csv_file = os.path.join(cwd, f'results/predictions{month}.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["model_name", "time", "actual_O3", "predicted_O3", "actual_NO2", "predicted_NO2"])
            writer.writerows(rows)


def train(processed, model_name, epochs):
    """
    Trains and returns the model
    """

    X_train, y_train, _, X_val, y_val, _, X_test, y_test, _ = processed

    if model_name == 'lr':
        model = LinearRegressor(X_train)
    elif model_name == 'mlp':
        model = MLPmodel(X_train)    # specify hyperparameters: n_steps_out, lr, dropout_rate, num_units1, num_units2
    elif model_name == 'lstm':
        model = LSTMmodel(X_train)
    else:
        print("Incorrect model's name.")

    if model is not None:
        model.train(X_train, y_train, X_val, y_val, epochs=epochs)
        print(f"{model_name} finished training.")

    return model.get_compiled_model()


def calculate_emissions():
    """
    Function for calculating each model's emissions from the emmisions.csv
    """
    data = pd.read_csv('results/emissions.csv')
    data['pollutant'] = ['O3', 'O3', 'O3', 'NO2', 'NO2', 'NO2']
    data['model'] = ['lr', 'mlp', 'lstm', 'lr', 'mlp', 'lstm']

    summary = data.groupby(['pollutant', 'model']).agg({
        'energy_consumed': 'sum',
        'emissions': 'sum'
    }).reset_index()

    print("\nSummary of Energy Consumption and CO₂ Emissions:")
    for _, row in summary.iterrows():
        print(f"Pollutant: {row['pollutant']}, Model: {row['model']}, "
            f"Total Energy Consumed: {row['energy_consumed']:.6f} Wh, "
            f"Total Emissions: {row['emissions']:.6f} kg CO₂")


def main():
    # NOTE: this holds the functionality for lab 2 (dashboard) and does not
    # save the trained model, fitted pca and scalers (for that see deployment/hugging_face_deployment.ipynb)

    n_steps_predict = 480
    months = [4, 7, 10, 12]

    # O3
    pollutant = 'O3'
    features = ['MWS', 'WG', 'T', 'TD', 'SD', 'GR']
    processed, _, o3_scaler, _ = prepare_data(pollutant, features)

    true_O3, lin_O3, _ = predict_pollutant(processed, n_steps_predict, months, 'lr', 15)
    true_O3, mlp_O3, _ = predict_pollutant(processed, n_steps_predict, months, 'mlp', 30)
    true_O3, lstm_O3, _ = predict_pollutant(processed, n_steps_predict, months, 'lstm', 30)

    # NO2
    pollutant = 'NO2'
    processed, _, no2_scaler, _ = prepare_data(pollutant, features)

    true_NO2, lin_NO2, _ = predict_pollutant(processed, n_steps_predict, months, 'lr', 15)
    true_NO2, mlp_NO2, _ = predict_pollutant(processed, n_steps_predict, months, 'mlp', 30)
    true_NO2, lstm_NO2, _ = predict_pollutant(processed, n_steps_predict, months, 'lstm', 30)

    # prepare data for dashboard
    prepare_dashboard_performance(lin_O3, mlp_O3, lstm_O3, lin_NO2, mlp_NO2, lstm_NO2)
    prepare_dashboard_predictions(n_steps_predict, true_O3, lin_O3, mlp_O3, lstm_O3, true_NO2, lin_NO2, mlp_NO2, lstm_NO2, o3_scaler, no2_scaler, months)


if __name__ == '__main__':
    main()
    # calculate_emissions()
