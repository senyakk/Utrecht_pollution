# Utrecht Air Pollution Forecast

---

A machine-learning pipeline to predict future air pollution levels in Utrecht. This is a system for predicting NO2 and O3 levels in Utrecht over a three-day period. Pollutant prediction is based on forecasted meteorological variables. The deployed MLP model can be accessed on [HuggingFace](https://huggingface.co/spaces/MLINPrediction/pollution_prediction) and features a user-friendly interface that provides accessible predictions, along with an alert mechanism to notify users of unusual or extreme conditions.

---

## Project Structure

```
├── **data/**: Folder with the datasets.
│   ├── **raw/**: Folder with the raw data.
│   ├── **processed/**: Folder with the processed data.
├── **src/**: Folder with the code.
│   ├── **data/**: Folder with the data pipeline.
│   ├── **features/**: Folder with the feature engineering pipeline.
│   ├── **models/**: Folder with the models.
│   ├── **utils/**: Folder with utility functions.
├──- **notebooks/**: Folder with the Jupyter notebooks.
├──- **results/**: Folder with the results.
├──- **tests/**: Folder with the tests.
├──- **deployment/**: Folder with the deployment code.
├──- **logs/**: Folder with the logs.
├──- **config/**: Folder with the configuration files.
├──- **README.md**: File with instructions.
├── **env/**: Python environment for the project. Activate with ```conda activate env/``` or ```pip install -r requirements``` to install it manually.
├── **requirements.txt**: File with the dependencies.
```


