import pandas as pd
import os
from dataset import DataSet


class MeteoDataset(DataSet):
    '''
    This class is inteded for loading weather features dataset for developing the model
    '''
    def __init__(self, variable:str) -> None:
        super().__init__(variable)

        if len(os.listdir(self._datasetdir)) != 1:
            raise ValueError(f"The directory {self._datasetdir} has more than one file.")

        self._datafile = self._datafiles[0]
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        '''
        Clean the data.
        '''

        self._dataframe = pd.read_csv(self._datafile, skiprows=20)
        self._filter_data()

    def _filter_data(self) -> None:
        '''
        Strip column names, convert hour format.
        '''
        self._dataframe.columns = self._dataframe.columns.str.strip()
        self._dataframe['HH'] = self._dataframe['HH'].astype(str).str.zfill(2)

        column_renames = {
            'DD': 'MWD',
            'FH': 'MWS',
            'FF': 'MWS10',
            'FX': 'WG',
            'SQ': 'SD',
            'Q': 'GR',
        }

        self._dataframe = self._dataframe.loc[:, ['YYYYMMDD', 'HH', 'DD', 'FH', 'FF', 'FX', 'T', 'TD', 'SQ', 'Q', 'P']]
        self._dataframe = self._dataframe.rename(columns=column_renames)
        self._dataframe['YYYYMMDD'] = self._dataframe['YYYYMMDD'].astype(int)
        self._dataframe['HH'] = self._dataframe['HH'].astype(int)
