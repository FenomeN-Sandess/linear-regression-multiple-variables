import numpy as np
import pandas as pd

def normalEqn(features: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Вычисляет параметры линейной регрессии через нормальное уравнение."""
    X = np.c_[np.ones(len(features)), features.values]
    y_vec = y.values.reshape(-1, 1)
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y_vec
    return pd.Series(theta.flatten())