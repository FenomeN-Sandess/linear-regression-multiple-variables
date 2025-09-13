import numpy as np
import pandas as pd

def gradientDescentMulti(theta:np.ndarray, features:pd.DataFrame, target:pd.Series,  al: float):
    feature1 = features.iloc[:, 0]
    feature2 = features.iloc[:, 1]
    h = theta[0] + theta[1]*feature1 + theta[2]*feature2 - target
    m = len(features)
    new_theta0 = theta[0] - al/m * sum(h)
    new_theta1 = theta[0] - al/m * sum(h*feature1)
    new_theta2 = theta[0] - al/m * sum(h*feature2)
    return np.array([new_theta0, new_theta1, new_theta2])