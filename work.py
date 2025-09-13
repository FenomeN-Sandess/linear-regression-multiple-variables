import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from methods.predict import predict
from methods.featureNormalize import featureNormalize

def work():\

    X_new = np.array(
        [float(input("Введите скорость оборота двигателя:\n>>")), float(input("Введите кол-во скоростей:\n>>"))])
    data = pd.read_pickle("data/model.pkl")

    normalize_X_new = featureNormalize("zscore", X_new, data["statictics"])
    print(
        f"Стоимость трактора предположительно составляет:\nПо данным модели, полученной аналитически:  {predict(data["an_theta"], X_new):.4f}\nПо данным модели, полученной через градиентный спуск: {predict(data["theta"], normalize_X_new):.4f}")


if __name__ == "__main__":
    work()
