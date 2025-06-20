import pandas as pd
from dataset_loader import DataSetLoader
from meteo_dataset import MeteoDataset
from pollutant_dataset import PollutantDataset

def main():
    """
    Loads both datasets for O3 and NO2 pollutants respectively.
    Then, loads the dataset with the weather features.
    Combines three datasets into one for model training.
    """
    loader = DataSetLoader()

    O3_dataset = PollutantDataset(variable="O3")
    df_O3 = O3_dataset.get_df()
    loader(O3_dataset, "data/processed/O3.csv")

    NO2_dataset = PollutantDataset(variable="NO2")
    df_NO2 = NO2_dataset.get_df()
    loader(NO2_dataset, "data/processed/NO2.csv")

    meteo_dataset = MeteoDataset(variable="meteo")
    df_meteo = meteo_dataset.get_df()
    loader(meteo_dataset, "data/processed/meteo.csv")

    combined = pd.concat([df_meteo, df_NO2[['NO2']], df_O3[['O3']]], axis=1)
    print(combined.head(10))
    loader(combined, "data/processed/combined.csv")



if __name__ == "__main__":
    main()
