import pandas as pd


# Нормализация признаков
def featureNormalize(method: str, features: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    if method == "zscore":
        return (features - st["mean"]) / st["std"]
    elif method == "center":
        return (features - st["mean"]) / (st["max"] - st["min"])
    elif method == "max":
        return features / st["max"]
    else:
        raise ValueError("Выберете правильный метод (zscore, center, max)")
