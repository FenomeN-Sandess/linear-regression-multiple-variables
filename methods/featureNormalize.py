import pandas as pd


# Нормализация признаков
def featureNormalize(method: str, features: pd.DataFrame) -> pd.DataFrame:
    if method == "zscore":
        features_mean = features.mean()
        features_std = features.std()
        return (features - features_mean) / features_std
    elif method == "center":
        features_mean = features.mean()
        features_max = features.max()
        features_min = features.min()
        return (features - features_mean) / (features_max - features_min)
    elif method == "max":
        features_max = features.max()
        return features / features_max
    else:
        raise ValueError("Выберете правильный метод (zscore, center, max)")
