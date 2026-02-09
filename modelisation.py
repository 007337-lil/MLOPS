import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor


def test_model(df, features, debut_test, params={}, model_name):
    """
    Docstring for test_model
    
    :param df: Description
    :param features: Description
    """
    df = df.loc[df['year'] >= 2016]
    X, y = df[features], df[['con_bru_gaz_tot']]
    X_train = X.loc[X.index.normalize() < pd.to_datetime(debut_test, format='%d/%m/%Y')]
    X_test = X.loc[X.index.normalize() >= pd.to_datetime(debut_test, format='%d/%m/%Y')]
    y_train = y.loc[y.index.normalize() < pd.to_datetime(debut_test, format='%d/%m/%Y')]
    y_test = y.loc[y.index.normalize() >= pd.to_datetime(debut_test, format='%d/%m/%Y')]

    if model_name == 'xgboost':
        model = XGBRegressor(
            random_state = 42
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importances = pd.Series(model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=False)

    df_verif = y_test.copy()
    df_verif['y_pred'] = y_pred

    return rmse, mae, r2, importances, df_verif