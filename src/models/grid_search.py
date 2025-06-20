from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor

def grid_search(model, X_train, y_train, cv=3):

    # param_grid = {
    #     'model__lr': [0.00001],
    #     'model__num_units1': [200, 120, 64],
    #     'model__num_units2': [64, 32],
    #     'model__dropout_rate': [0.2],
    #     'batch_size': [32],
    #     'epochs': [40]
    # }

    param_grid = {
    'model__l1_penalty': [0.001, 0.01, 0.1],
    'model__l2_penalty': [0.001, 0.01, 0.1],
    'model__lr': [0.001, 0.01, 0.1],
    }

    model = KerasRegressor(build_fn = model, verbose = 0)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, verbose=1, n_jobs=-1)

    grid_result = grid.fit(X_train, y_train)

    print(f'Best Score: {grid_result.best_score_}')
    print(f'Best Parameters: {grid_result.best_params_}')