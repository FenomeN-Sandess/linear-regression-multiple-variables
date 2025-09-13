import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from methods.gradientDestentMulti import gradientDescentMulti
from methods.computeCostMulti import computeCostMulti
from methods.featureNormalize import featureNormalize
from methods.normalEqn import normalEqn


def main():
    data = pd.read_csv("data/ex1data2.txt", delimiter=",", names=["engine_speed", "amount_gears", "price"])

    features = pd.DataFrame({"engine": data["engine_speed"], "gears": data["amount_gears"]})
    price = pd.Series(data["price"])

    # Статистические характеристики обучающей выборки для вычисления нормализации
    features_mean = features.mean()
    features_std = features.std()
    features_max = features.max()
    features_min = features.min()

    statictics = pd.DataFrame({"mean": features_mean, "std": features_std, "max": features_max, "min": features_min})

    normalizeFeatures = featureNormalize("zscore", features, statictics)
    theta = np.ones(3)
    alphas = np.array([i * 0.01 for i in range(1, 11)])
    alphaTime = {alpha: [theta, 0] for alpha in alphas}
    epsilon = 10 ** (-6)

    for a in alphas:
        start = time.perf_counter()
        while True:
            new_theta = gradientDescentMulti(theta, normalizeFeatures, price, a)
            difference_epsilon = computeCostMulti(new_theta, normalizeFeatures, price) - computeCostMulti(theta,
                                                                                                          normalizeFeatures,
                                                                                                          price)
            if abs(difference_epsilon) < epsilon:
                break

            theta = new_theta

        end = time.perf_counter()
        alphaTime[a][0] = theta
        alphaTime[a][1] = end - start

    alphaTime = min(alphaTime.items(), key=lambda x: x[1][1])
    theta = alphaTime[1][0]
    print(f"Сходимость функции достигнута со значением alpha = {alphaTime[0]} за минимальное время {alphaTime[1][1]}")
    print(
        f"Полученные параметры через градиентный спуск:\ntheta_0 = {theta[0]}\ntheta_1 = {theta[1]}\ntheta_2 = {theta[2]}")

    an_theta = normalEqn(features, price)
    print(
        f"Значения theta полученные аналитически:\ntheta_0 = {an_theta[0]}\ntheta_1 = {an_theta[1]}\ntheta_2 = {an_theta[2]}")

    data = {
        "theta": theta,
        "an_theta": an_theta,
        "statictics": statictics,
    }

    pd.to_pickle(data, "data/model.pkl")

    #np.savez("data/theta_saves.npz", theta=theta, an_theta=an_theta, statictics=statictics)


if __name__ == '__main__':
    main()
