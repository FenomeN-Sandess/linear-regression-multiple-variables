import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from methods.featureNormalize import featureNormalize
from methods.gradientDestentMulti import gradientDescentMulti
from methods.computeCostMulti import computeCostMulti
from methods.featureNormalize import featureNormalize


def main():
    data = pd.read_csv("data/ex1data2.txt", delimiter=",", names=["engine_speed", "amount_gears", "price"])

    features = pd.DataFrame({"engine": data["engine_speed"], "gears": data["amount_gears"]})
    price = pd.Series(data["price"])


    normalizeFeatures = featureNormalize("zscore", features)
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
    print(f"Полученные параметры:\ntheta_0 = {theta[0]}\ntheta_1 = {theta[1]}\ntheta_2 = {theta[2]}")


if __name__ == '__main__':
    main()
