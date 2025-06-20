import pandas as pd
from dataset import DataSet


class PollutantDataset(DataSet):
    '''
    This class is inteded for loading pollutant historical dataset for developing the model
    '''
    def __init__(self, variable:str, n_files = None, station = 'NL10643') -> None:
        '''
        n_files: specifies loading the most recent n files
        '''
        super().__init__(variable)
        
        if n_files is not None:
            if not isinstance(n_files, int) or n_files <= 0:
                raise ValueError("number of recent files to load must be a positive integer or None")
            self._datafiles = self._datafiles[-n_files:]
        
        self._station = station
        self._preprocess_data()
    
    def _preprocess_data(self):
        '''
        Preprocessing:
            - change the date format + variable name
            - keep the specified station data
            - take care of NULL values
            - combine the data frames into a single one
        '''

        dfs = []
        for file in self._datafiles:
            print(file)
            pollutant_df = pd.read_csv(file, encoding = 'ISO-8859-15', sep =';', skiprows = 9)
            dfs.append(pollutant_df)
        
        self._dataframe = pd.concat(dfs, axis=0)
        self._dataframe = self._dataframe.reset_index(drop = True)
        self._filter_data()

    def _filter_data(self) -> None:
        '''
        Changes the date format from ISO-8859-15 to YYYYMMDD + HH,
        renames the station column to pollutant name,
        returns the specified station data only.
        '''
        self._change_dateformat()
        self._dataframe = self._dataframe[['YYYYMMDD', 'HH', self._station]].rename(columns={self._station: self._variable})
        self._round_measurement()
        self._impute_null()

        self._dataframe['YYYYMMDD'] = self._dataframe['YYYYMMDD'].astype(int)
        self._dataframe['HH'] = self._dataframe['HH'].astype(int)

    def _change_dateformat(self) -> None:
        self._dataframe['Einddatumtijd'] = pd.to_datetime(self._dataframe['Einddatumtijd'], format='%Y%m%d %H:%M')
        self._dataframe.loc[:, 'YYYYMMDD'] = self._dataframe['Einddatumtijd'].dt.strftime('%Y%m%d')
        self._dataframe.loc[:, 'HH'] = self._dataframe['Einddatumtijd'].dt.hour.astype(str)

    def _round_measurement(self) -> None:
        self._dataframe[self._variable] = self._dataframe[self._variable].round(2)

    def _impute_null(self) -> None:
        i = 0
        while i < len(self._dataframe):
            if (pd.isna(self._dataframe.iloc[i][self._variable])):
                prev_idx = i - 1
                while i < len(self._dataframe) and pd.isna(self._dataframe.iloc[i][self._variable]):
                    i += 1
                next_idx = i
                gap_size = next_idx - prev_idx - 1
                if gap_size < 8:
                    self._lin_interpolation(prev_idx, next_idx, gap_size)
                else:
                    i += 1
            else:
                i += 1
        self._avg_days()
        self._impute_lastResort()

    def _lin_interpolation(self, prev_idx:int, next_idx:int, gap_size:int) -> None:
        prev_val = self._dataframe.iloc[prev_idx][self._variable]
        next_val = self._dataframe.iloc[next_idx][self._variable]

        delta = (next_val - prev_val) / (gap_size + 1)
        
        # X(n) = X(n-1) + delta
        for i in range(prev_idx+1, next_idx):
            imputed_value = self._dataframe.iloc[i - 1][self._variable] + delta
            self._dataframe.at[i, self._variable] = round(imputed_value, ndigits=2)

    def _avg_days(self) -> None:
        nulls = self._dataframe[self._variable].isna()
        for i in self._dataframe.index[nulls]:
            if i >= 24 and i + 24 < len(self):
                prev = self._dataframe.iloc[i - 24][self._variable]
                next = self._dataframe.iloc[i + 24][self._variable]
                if pd.notna(prev) and pd.notna(next):
                    self._dataframe.at[i, self._variable] = (prev + next) / 2
    
    def _impute_lastResort(self) -> None:
        nulls = self._dataframe[self._variable].isna()
        for i in self._dataframe.index[nulls]:
            if i >= 96 and i + 96 < len(self):
                prev = self._dataframe.iloc[i - 96][self._variable]
                next = self._dataframe.iloc[i + 96][self._variable]
                if pd.notna(prev) and pd.notna(next):
                    self._dataframe.at[i, self._variable] = (prev + next) / 2
