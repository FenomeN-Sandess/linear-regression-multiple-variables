import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from methods.featureNormalize import featureNormalize
from methods.gradientDestentMulti import gradientDescentMulti
from methods.computeCostMulti import computeCostMulti
from methods.featureNormalize import featureNormalize

def main():
    data = pd.read_csv("data/ex1data2.txt", delimiter=",", names = ["engine_speed", "amount_gears", "price"])

    features = pd.DataFrame({"engine": data["engine_speed"], "gears": data["amount_gears"]})
    price = pd.Series(data["price"])

    normalizeFeatures = featureNormalize(features)

    theta = pd.Series(np.ones(3))
    al = 0.1
    epsilon = 10**(-6)


    while True:
        new_theta = gradientDescentMulti(theta, normalizeFeatures, price, al)
        difference_epsilon = computeCostMulti(new_theta, normalizeFeatures, price) - computeCostMulti(theta, normalizeFeatures, price)
        print(f"Разность: {difference_epsilon}")
        if abs(difference_epsilon) < epsilon:
            break

        theta = new_theta

    print(new_theta)

if __name__ == '__main__':
    main()




