import pandas as pd
import numpy as np
import os
from .sliding_window import split_sequences_multi_output
from .featureEng import FeatureEngineering
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from typing import List

PCA_COMPONENTS = 15

class FeatureEngineeringTraining(FeatureEngineering):
    '''
    Performs feature engineering, including extracting lagged and rolling
    window features, and processing the data for machine learning models.
    '''
    def __init__(self, file_name):
        super().__init__(file_name)
    
    def process(self, train_test:int, train_val:int, n_steps:int, n_steps_out:int):
        '''
        Splits the dataset into training, validation, and test sets, scales the data, 
        and formats the sequences for input into time series models.
        '''
        X_train, y_train, X_val, y_val, X_test, y_test = self._split(train_test, train_val)
        X_train_scaled, X_val_scaled, X_test_scaled = self._scale(X_train, X_val, X_test, are_features = True)
        y_train_scaled, y_val_scaled, y_test_scaled = self._scale(y_train, y_val, y_test, are_features = False)

        X_train_scaled, X_val_scaled, X_test_scaled = self._PCA(X_train_scaled, X_val_scaled, X_test_scaled, n_components=PCA_COMPONENTS)

        X_train_seq, y_train_seq_target, y_train_seq_features = split_sequences_multi_output(X_train_scaled, y_train_scaled, n_steps, n_steps_out)
        X_val_seq, y_val_seq_target, y_val_seq_features = split_sequences_multi_output(X_val_scaled, y_val_scaled, n_steps, n_steps_out)
        X_test_seq, y_test_seq_target, y_test_seq_features = split_sequences_multi_output(X_test_scaled, y_test_scaled, n_steps, n_steps_out)

        return [X_train_seq, y_train_seq_target, y_train_seq_features, X_val_seq, y_val_seq_target, y_val_seq_features, X_test_seq, y_test_seq_target, y_test_seq_features]

    def _split(self, train_test:int, train_val:int):
        '''
        Splits the data into training, validation, and test sets based on the
        provided proportions.
        '''
        split_index = int(train_test * len(self._X))

        X_train, X_test = self._X[:split_index], self._X[split_index:]
        y_train, y_test = self._y[:split_index], self._y[split_index:]

        split_index = int(train_val * len(X_train))
        X_train, X_val = X_train[:split_index], X_train[split_index:]
        y_train, y_val = y_train[:split_index], y_train[split_index:]

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _scale(self, train, val, test, are_features):
        '''
        Scales the training, validation, and test sets using MinMax scaling.
        '''
        if are_features:
            self._features_fit_scaler = MinMaxScaler()
            print(f"Scaler is fitted on the size: {train.shape}")
            train_scaled = self._features_fit_scaler.fit_transform(train)
            val_scaled = self._features_fit_scaler.transform(val)
            test_scaled = self._features_fit_scaler.transform(test)
        else:
            self._target_fit_scaler = MinMaxScaler()
            print(f"Scaler is fitted on the size: {train.shape}")
            train_scaled = self._target_fit_scaler.fit_transform(train)
            val_scaled = self._target_fit_scaler.transform(val)
            test_scaled = self._target_fit_scaler.transform(test)
            
        return train_scaled, val_scaled, test_scaled
    

    def _PCA(self, X_train_scaled, X_val_scaled, X_test_scaled, n_components=PCA_COMPONENTS):
        '''
        Reduces dimentionality of data. Uses PCA_COMPONENTS global variable
        '''
        self._fit_pca = PCA(n_components=n_components)
        X_train_pca = self._fit_pca.fit_transform(X_train_scaled)
        X_val_pca = self._fit_pca.transform(X_val_scaled)
        X_test_pca = self._fit_pca.transform(X_test_scaled)

        return X_train_pca, X_val_pca, X_test_pca
    
    def get_fitted_PCA(self):
        '''
        Returns PCA fitted on the training features
        '''
        return self._fit_pca
    
    def get_fitted_scaler(self, are_features):
        '''
        Returns fitted scaler
        are_features = True, returns scaler fitted on training features
        are_features = False, returns scaler fitted on training targets.
        '''
        if are_features:
            return self._features_fit_scaler
        else:
            return self._target_fit_scaler
