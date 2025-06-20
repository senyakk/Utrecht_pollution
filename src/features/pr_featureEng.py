import pandas as pd
import numpy as np
import os
from .sliding_window import split_sequences_multi_output
from .featureEng import FeatureEngineering
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from typing import List


class FeatureEngineeringPredict(FeatureEngineering):
    '''
    Performs feature engineering, including extracting lagged and rolling
    window features, and processing the data for machine learning models.
    '''
    def __init__(self, file_name, fit_pca:PCA, fit_scaler:MinMaxScaler):
        super().__init__(file_name)
        self._fit_pca = fit_pca
        self._fit_scaler = fit_scaler
    
    
    def process(self, n_steps:int, n_steps_out:int):
        '''
        scales the data, 
        formats the sequences for input into time series models.
        '''
        self._scale()
        self._PCA()

        X_seq, _, _ = split_sequences_multi_output(self._X, None, n_steps, n_steps_out)

        return X_seq

    def _scale(self):
        '''
        Scales the training, validation, and test sets using MinMax scaling.
        '''
        self._X = self._fit_scaler.transform(self._X)


    def _PCA(self):
        '''
        Reduces dimentionality of data
        '''
        self._X = self._fit_pca.transform(self._X)
