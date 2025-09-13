import pandas as pd
import numpy as np


def computeCostMulti(theta: np.ndarray, features: pd.DataFrame, y: pd.Series) -> float:
    feature1 = features.iloc[:, 0]
    feature2 = features.iloc[:, 1]
    difference = theta[0] + theta[1] * feature1 + theta[2] * feature2 - y
    len_array = len(features)
    return (1 / (2 * len_array)) * sum(difference ** 2)
