import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

def get_dataset(df_name):
    df = pd.read_csv(df_name)
    return pd.DataFrame(df['X']), df['y']


def test_dataset():
    X_train, y_train = get_dataset('df1.csv')
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    for i in range(2, 5):
        X_test, y_test = get_dataset('df' + str(i) + '.csv')
        assert mse(model.predict(X_test), y_test) < 1, f"Датасет 'df{str(i)}.csv' зашумлён!"
