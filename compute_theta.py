import pandas as pd
import numpy as np

from methods.normalEqn import normalEqn


def gradient_descent_theta(features: pd.DataFrame, y: pd.Series, alpha: float = 0.01,
                           iterations: int = 1500) -> pd.Series:
    """Вычисляет вектор параметров методом градиентного спуска.
    Возвращает параметры в масштабе исходных признаков."""
    mu = features.mean()
    sigma = features.std()
    norm_features = (features - mu) / sigma
    X = np.c_[np.ones(len(norm_features)), norm_features.values]
    y_vec = y.values.reshape(-1, 1)
    theta = np.zeros((X.shape[1], 1))
    m = len(X)
    for _ in range(iterations):
        theta = theta - alpha / m * (X.T @ (X @ theta - y_vec))

    t0 = theta[0] - (theta[1] * mu.iloc[0] / sigma.iloc[0]) - (theta[2] * mu.iloc[1] / sigma.iloc[1])
    t1 = theta[1] / sigma.iloc[0]
    t2 = theta[2] / sigma.iloc[1]
    return pd.Series([t0, t1, t2]).astype(float)


def compute_theta():
    data = pd.read_csv("data/ex1data2.txt", delimiter=",", names=["size", "rooms", "price"])
    features = data[["size", "rooms"]]
    y = data["price"]

    theta_normal = normalEqn(features, y)
    print("Theta через нормальное уравнение:", theta_normal.tolist())

    theta_gd = gradient_descent_theta(features, y)
    print("Theta после градиентного спуска:", theta_gd.tolist())


