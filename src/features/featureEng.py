import pandas as pd
import numpy as np
import os
from typing import List
from abc import ABC, abstractmethod


class FeatureEngineering(ABC):
    '''
    Performs feature engineering, including extracting lagged and rolling
    window features, and processing the data for machine learning models.
    '''
    def __init__(self, file_name):

        if os.path.basename(os.getcwd()) == 'notebooks' or os.path.basename(os.getcwd()) == 'deployment':
            os.chdir('..')

        dataset_path = f'data/processed/{file_name}'
        self._data = pd.read_csv(dataset_path)
    
    def extract_features(self, selected_features:List[str], target:List[str],
                        lags:List[int]=[24,48], rolling_windows:List[int]=[3,6,12,24], drop_ogs=0):
        
        '''
        Extracts features from the time series data, including lagged and rolling 
        window features, and updates the feature set.
        '''

        if not all(f in self._data.columns for f in selected_features):
            raise ValueError(f"Select features in {self._data.columns}")

        self._features = []

        if lags is not None and target is not None:
            self._lagging(lags, target)
        
        if rolling_windows is not None:
            self._rolling_windows(rolling_windows, selected_features)
    
        # Drop any rows with missing values (introduced by lagging)
        self._data = self._data.dropna()

        # If not dropping the original features, include them in the feature list
        if not drop_ogs:
            self._features += selected_features
        
        # Feature and target values
        self._X = self._data[self._features].values

        # Uncomment if you want to save data after applying "rolling window"
        # df = self._data[self._features]
        # df.to_csv('data/processed/fe_combined.csv', index=False)

        if target is not None:
            self._y = self._data[target].values
    
    def _lagging(self, lags:List[int], target:List[str]):
        '''
        Creates lagged features for the target variables based on the specified
        lag intervals.
        '''
        for lag in lags:
            for feature in target:
                # Create lagged features by shifting the target variable by the
                # specified lag amount (e.g. 24 hours back)
                self._data[f'{feature}_lag{lag}'] = self._data[feature].shift(lag)
                self._features.append(f'{feature}_lag{lag}')
    
    def _rolling_windows(self, rolling_windows:List[int], features:List[str]):
        '''
        Computes rolling window features for the specified features based on the 
        provided window sizes.
        '''
        for window in rolling_windows:
            for feature in features:
                self._data[f'{feature}_rolling{window}'] = self._data[feature].rolling(window=window).mean()
                self._features.append(f'{feature}_rolling{window}')
    
    @abstractmethod
    def process(self):
        '''
        (Splits the dataset into training, validation, and test sets),
        scales the data, formats the sequences for input into time series models.
        '''
        pass

    @abstractmethod
    def _scale(self):
        '''
        Scales the training, validation, and test sets using MinMax scaling.
        '''
        pass
    
    @abstractmethod
    def _PCA(self):
        '''
        Reduces dimentionality of data
        '''
        pass
