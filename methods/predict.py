import numpy as np

def predict(theta: np.ndarray, X_new:np.ndarray) -> float:
    return float(theta[0] + theta[1] * X_new[0] + theta[2] * X_new[1])