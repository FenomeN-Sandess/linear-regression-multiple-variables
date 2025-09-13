import numpy as np
import pandas as pd

def normalEqn(X: pd.DataFrame, y: pd.Series):
    X = np.hstack([np.ones((X.shape[0], 1)),X]) #+ столбец единиц для theta_0
    return np.linalg.inv(X.T@X)@X.T@y # Формула (X^T * X)^(-1) * X^T * y

