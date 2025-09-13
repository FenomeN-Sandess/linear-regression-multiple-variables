import pandas as pd

def computeCostMulti(theta:pd.Series, features: pd.DataFrame, y:pd.Series)->float:
    feature1 = features.iloc[:, 0]
    feature2 = features.iloc[:, 1]
    difference = theta.iloc[0] + theta.iloc[1] * feature1 + theta.iloc[2] * feature2 - y
    len_array = len(features)
    return (1/(2*len_array))*sum(difference**2)

