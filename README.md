# 🚦 Utrecht Air Pollution Forecast

---

A machine-learning pipeline to predict future air pollution levels in Utrecht. This is a system for predicting NO2 and O3 levels in Utrecht over a three-day period. Pollutant prediction is based on forecasted meteorological variables. The deployed MLP model can be accessed on [HuggingFace](https://huggingface.co/spaces/MLINPrediction/pollution_prediction) and features a user-friendly interface that provides accessible predictions, along with an alert mechanism to notify users of unusual or extreme conditions.

The data for prediction is automatically fetched from the [Royal Netherlands Meteorological Institute](https://open-meteo.com/en/docs/knmi-api#latitude=52.11&longitude=5.1806&hourly=temperature_2m,dew_point_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,sunshine_duration,global_tilted_irradiance&daily=weather_code) API.

<div align="center">
  <img width="560" alt="Screenshot 2025-06-20 at 10 46 10" src="https://github.com/user-attachments/assets/5ff23255-5e27-4a84-8f4a-2f4841750795" />
</div>

---

## Project Structure

```
├── data/: Folder with the datasets.
│   ├──raw/: Folder with the raw data.
│   ├── processed/: Folder with the processed data.
├── src/: Folder with the code.
│   ├── data/: Folder with the data pipeline.
│   ├── features/: Folder with the feature engineering pipeline.
│   ├── models/: Folder with the models.
│   ├── utils/: Folder with utility functions.
├──- notebooks/: Folder with the Jupyter notebooks.
├──- results/: Folder with the results.
├──- tests/: Folder with the tests.
├──- deployment/: Folder with the deployment code.
├──- logs/: Folder with the logs.
├──- config/: Folder with the configuration files.
├──- README.md: File with instructions.
├── env/: Python environment for the project. 
├── requirements.txt: File with the dependencies.
```
---

## Code Instructions

Activate environment with 
```bash 
conda activate env/
``` 

Or alternatively create new environment and activate iit manually by 
```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```


## Data preprocessing (not needed at this step)

The data preprocessing can be run from ```src/data/main.py```. It reads the datasets
from data/raw and inserts the processed data into ```data/processed```. The feature
engineering is run from ```src/features/feature_eng.py``` which loads the
preprocessed datasets and saves the pca transformed data into ```data/processed```.

## Model training and data preparation for the dashboard

Running 
```bash
python -m src.models.main
``` 

trains 6 different models (O3 and NO2
linear regressor, O3 and NO2 LSTM, O3 and NO2 MLP). It saves all the data
shown in the dashboard under "results". The number of time steps to predict
can be specified under "n_steps_predict", and the month to predict under
"month". The data under results currently shows the model's predictions for
April. 

The dashboard is run via 
```bash
streamlit run src/utils/dashboard.py
```


