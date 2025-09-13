import pandas as pd
import numpy as np
import pytest

from methods.computeCostMulti import computeCostMulti
from methods.featureNormalize import featureNormalize
from methods.gradientDestentMulti import gradientDescentMulti


# Тесты функции computeCostMulti с разными наборами данных

def test_compute_cost_multi_case1():
    """Проверка стоимости на первом наборе данных"""
    features = pd.DataFrame({"f1": [1, 3], "f2": [2, 4]})
    y = pd.Series([6, 10])
    theta = pd.Series([0, 1, 2])
    expected = 0.5
    result = computeCostMulti(theta, features, y)
    assert result == pytest.approx(expected)


def test_compute_cost_multi_case2():
    """Проверка стоимости на втором наборе данных"""
    features = pd.DataFrame({"f1": [0, 1], "f2": [0, 1]})
    y = pd.Series([0, 1])
    theta = pd.Series([0, 0, 0])
    expected = 0.25
    result = computeCostMulti(theta, features, y)
    assert result == pytest.approx(expected)


# Тесты функции featureNormalize с разными наборами данных

def test_feature_normalize_case1():
    """Нормализация первого набора признаков"""
    features = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    norm = featureNormalize(features)
    assert np.allclose(norm.mean(), 0)
    assert np.allclose(norm.std(ddof=1), 1)


def test_feature_normalize_case2():
    """Нормализация второго набора признаков"""
    features = pd.DataFrame({"a": [10, 20, 30], "b": [20, 30, 40]})
    norm = featureNormalize(features)
    assert np.allclose(norm.mean(), 0)
    assert np.allclose(norm.std(ddof=1), 1)


# Тесты функции gradientDescentMulti с разными наборами данных

def test_gradient_descent_multi_case1():
    """Градиентный спуск для простого набора данных"""
    features = pd.DataFrame({"f1": [1, 1], "f2": [1, 1]})
    target = pd.Series([2, 2])
    theta = pd.Series([0, 0, 0])
    alpha = 0.1
    expected = pd.Series([0.2, 0.2, 0.2])
    result = gradientDescentMulti(theta, features, target, alpha)
    assert np.allclose(result.values, expected.values)


def test_gradient_descent_multi_case2():
    """Градиентный спуск для другого набора данных"""
    features = pd.DataFrame({"f1": [1, 2], "f2": [1, 2]})
    target = pd.Series([3, 6])
    theta = pd.Series([1, 1, 1])
    alpha = 0.01
    expected = pd.Series([1.005, 1.01, 1.01])
    result = gradientDescentMulti(theta, features, target, alpha)
    assert np.allclose(result.values, expected.values)


# Вспомогательная функция для тестов общего алгоритма

def run_algorithm(features: pd.DataFrame, y: pd.Series, alpha: float, iterations: int) -> float:
    """Полный цикл нормализации и обучения"""
    norm_features = featureNormalize(features)
    theta = pd.Series(np.ones(3))
    for _ in range(iterations):
        theta = gradientDescentMulti(theta, norm_features, y, alpha)
    return computeCostMulti(theta, norm_features, y)


# Тесты общего алгоритма с разными наборами данных

def test_full_algorithm_case1():
    """Проверка уменьшения стоимости на первом наборе"""
    features = pd.DataFrame({"f1": [1, 3], "f2": [2, 4]})
    y = pd.Series([6, 10])
    alpha = 0.1
    initial_cost = computeCostMulti(pd.Series([1, 1, 1]), featureNormalize(features), y)
    final_cost = run_algorithm(features, y, alpha, 10)
    assert final_cost < initial_cost


def test_full_algorithm_case2():
    """Проверка уменьшения стоимости на втором наборе"""
    features = pd.DataFrame({"f1": [1, 0, 1], "f2": [0, 1, 1]})
    y = pd.Series([1, 1, 2])
    alpha = 0.01
    initial_cost = computeCostMulti(pd.Series([1, 1, 1]), featureNormalize(features), y)
    final_cost = run_algorithm(features, y, alpha, 20)
    assert final_cost < initial_cost
