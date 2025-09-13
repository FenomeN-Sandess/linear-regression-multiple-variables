import pandas as pd

# Нормализация признаков
def featureNormalize(features: pd.DataFrame)->pd.DataFrame:
    features_mean = features.mean()
    features_std = features.std()
    return (features - features_mean)/features_std

