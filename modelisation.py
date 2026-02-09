import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor


def test_model(df, features, debut_test, model_name, params={}):
    """
    Docstring for test_model
    
    :param df: Description
    :param features: Description
    :param debut_test: Description
    :param model_name: Description
    :param params: Description
    """
    df = df.loc[df['year'] >= 2016]
    X, y = df[features], df.iloc[:, -1]
    X_train = X.loc[X.index.normalize() < pd.to_datetime(debut_test, format='%d/%m/%Y')]
    X_test = X.loc[X.index.normalize() >= pd.to_datetime(debut_test, format='%d/%m/%Y')]
    y_train = y.loc[y.index.normalize() < pd.to_datetime(debut_test, format='%d/%m/%Y')]
    y_test = y.loc[y.index.normalize() >= pd.to_datetime(debut_test, format='%d/%m/%Y')]

    if model_name == 'xgboost':
        model = XGBRegressor(
            random_state = 42,
            eval_metric = 'rmse',
            enable_categorical = True
        )

    model.set_params(**params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importances = pd.Series(model.feature_importances_.round(3), index=features)
    importances = importances.sort_values(ascending=False)

    df_verif = pd.DataFrame(y_test)
    df_verif['y_pred'] = y_pred

    return rmse, mae, r2, importances, df_verif


def search_params(df, features, debut_test, model_name, param_grid):
    """
    Docstring for test_model
    
    :param df: Description
    :param features: Description
    :param debut_test: Description
    :param model_name: Description
    :param params: Description
    """
    df = df.loc[df['year'] >= 2016]
    X, y = df[features], df.iloc[:, -1]
    X_train = X.loc[X.index.normalize() < pd.to_datetime(debut_test, format='%d/%m/%Y')]
    X_test = X.loc[X.index.normalize() >= pd.to_datetime(debut_test, format='%d/%m/%Y')]
    y_train = y.loc[y.index.normalize() < pd.to_datetime(debut_test, format='%d/%m/%Y')]
    y_test = y.loc[y.index.normalize() >= pd.to_datetime(debut_test, format='%d/%m/%Y')]

    tscv = TimeSeriesSplit(n_splits=3) 

    if model_name == 'xgboost':
        model = XGBRegressor(
            random_state = 42,
            eval_metric = 'rmse',
            enable_categorical = True
        )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,    
        verbose=2,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    y_pred = grid.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    df_verif = y_test.copy()
    df_verif['y_pred'] = y_pred

    return rmse, mae, r2, df_verif, best_params