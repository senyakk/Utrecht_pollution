# Air Pollution Prediction in Utrecht

--

This project focuses on building a scalable automated air pollution forecasting system to predict NO\textsubscript{2} and O\textsubscript{3} levels in Utrecht over a three-day period. Pollutant prediction is based on forecasted meteorological variables. The deployed MLP model features a user-friendly interface that provides accessible predictions, along with an alert mechanism to notify users of unusual or extreme conditions.

--

The basic structure of the repository is as follows, from the root folder:

- **data/**: Folder with the datasets.
    - **raw/**: Folder with the raw data.
    - **processed/**: Folder with the processed data.

- **src/**: Folder with the code.
    - **data/**: Folder with the data pipeline.
    - **features/**: Folder with the feature engineering pipeline.
    - **models/**: Folder with the models.
    - **utils/**: Folder with utility functions.

- **notebooks/**: Folder with the Jupyter notebooks.

- **results/**: Folder with the results.

- **tests/**: Folder with the tests.

- **deployment/**: Folder with the deployment code.

- **logs/**: Folder with the logs.

- **config/**: Folder with the configuration files.

- **README.md**: File with instructions.

- **env/**: Python environment for the project. Activate with ```conda activate env/``` or ```pip install -r requirements``` to install it manually.

- **requirements.txt**: File with the dependencies.
