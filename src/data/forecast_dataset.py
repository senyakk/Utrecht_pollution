import pandas as pd
from dataset import DataSet
from dataset_loader import DataSetLoader

class ForecastMeteo(DataSet):
    '''
    This class is inteded for loading weather forecast dataset provided by public API
    '''
    def __init__(self, variable:str) -> None:
        super().__init__(variable)
        self._datafile = self._datafiles[0]
        self._dataframe = pd.read_csv(self._datafile)

        self._filter_data()

    def _filter_data(self) -> None:
        '''
        Clean the data.
        '''
        self._dataframe = self._dataframe.drop('date', axis=1)
        self._rename_reorder()       # reorder columns
        self._fix_number_representation()
        print(self._dataframe)

    def _change_date_column(self):
        """
        Converts the 'date' column into 'YYYYMMDD' and 'HH' columns.
        """
        self._dataframe['date'] = pd.to_datetime(self._dataframe['date'], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
        
        # create 'YYYYMMDD' and 'HH' columns
        self._dataframe['YYYYMMDD'] = self._dataframe['date'].dt.strftime('%Y%m%d').astype(int)
        self._dataframe['HH'] = self._dataframe['date'].dt.hour.astype(int)

        self._dataframe['YYYYMMDD'] = self._dataframe['YYYYMMDD'].astype(int)
        self._dataframe['HH'] = self._dataframe['HH'].astype(int)

        # shift the range from 0-23 to 1-24 to match our training data
        self._dataframe['HH'] = self._dataframe['HH'] + 1

        self._dataframe = self._dataframe.drop(columns=['date']) # drop the original 'date' column
    
    def _rename_reorder(self):
        self._dataframe.rename(columns={
            'wind_speed_10m': 'MWS',
            'wind_gusts_10m': 'WG',
            'sunshine_duration': 'SD',
            'global_tilted_irradiance': 'GR',
            'temperature_2m': 'T',
            'dew_point_2m': 'TD'
        }, inplace=True)

        # reorder columns
        self._dataframe = self._dataframe[['MWS', 'WG', 'T', 'TD', 'SD', 'GR']]

    def _fix_number_representation(self):
        """
        This method ensures consistent decimal precision and scales the columns appropriately.
        """

        # T and TD converted from degrees C to 0.1 degrees C
        self._dataframe['T'] = self._dataframe['T'] * 10
        self._dataframe['TD'] = self._dataframe['TD'] * 10

        # MWS and WG converted from m/s to 0.1 m/s
        self._dataframe['MWS'] = self._dataframe['MWS'] * 10
        self._dataframe['WG'] = self._dataframe['WG'] * 10

        # SD converted from s to 0.1 h
        self._dataframe['SD'] = self._dataframe['SD'] / 360

        # GR converted from W/m2 to J/cm2
        self._dataframe['GR'] = self._dataframe['GR'] * 0.36

        self._dataframe = self._dataframe.round(2)


if __name__ == "__main__":
    preprocess = ForecastMeteo(variable="forecast")
    loader = DataSetLoader()
    loader(preprocess.get_df(), "data/processed/forecast_example.csv")
