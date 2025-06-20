import os
import pandas as pd
import numpy as np
import csv
from abc import ABC, abstractmethod

class DataSet(ABC):
    '''
    Manage datasets of pollutants and meteorological data, in the format
    specified by KNMI.
    '''
    def __init__(self, variable) -> None:
        cwd = os.getcwd()
        dataset_dir = os.path.join(cwd, 'data/raw', variable)
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"The directory {dataset_dir} does not exist.")
        if not os.listdir(dataset_dir):
            raise ValueError(f"The directory {dataset_dir} is empty.")
        
        # if not all(os.path.isfile(os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir))):
        #     raise ValueError(f"The directory {dataset_dir} must only include files.")

        self._datafiles = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)])

        self._datasetdir = dataset_dir
        self._dataframe = None
        self._variable = variable

    def __getitem__(self, index:int):
        '''
        Allows its instances to use the [] (indexer) operators.
        '''
        if index not in range(0, len(self)):
            raise IndexError("Index out of bounds.")
        return self._dataframe.iloc[index]
    
    def __len__(self) -> int:
        '''
        Get the number of datapoints in the dataset.
        '''
        return len(self._dataframe)
    
    def get_df(self) -> pd.DataFrame:
        return self._dataframe

    @abstractmethod
    def _filter_data(self):
        '''
        Filters the dataset based on the variable(s) included.
        '''
        pass
