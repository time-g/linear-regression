#!/usr/bin/env python3
import numpy as np


def sse(y, yhat):
    """
    Sum of squared errors
    """
    n = y.shape[0]  # len of y
    error = (y - yhat) ** 2
    result = error.sum()
    return result


def mse(y, yhat):
    """
    Mean Squared Error
    """
    n = y.shape[0]  # len of y
    error = (y - yhat) ** 2
    result = error.sum() / n
    return result


def rmse(y, yhat):
    """
    Root Mean Square Error
    """
    n = y.shape[0]  # len of y
    error = (y - yhat) ** 2
    result = np.sqrt(error).sum() / n
    return result


def gradient_descent(x: np.array, y: np.array, eta, niter=800, cost=mse):
    """
    y = w0 + x w^T
    MSE(w)  = 1/n sum([yhat - y]^2)
    gradient of MSE(w) = 1/n [x^T (yhat - y)]
    """
    try:
        number_of_feature = x.shape[1] + 1
    except:
        number_of_feature = 2
        x = np.array(x)
        x.shape = (x.shape[0], 1)

    w = np.zeros(number_of_feature)
    costs = np.zeros(niter)
    n = x.shape[0]

    for i in range(niter):
        yhat = w[0] + x @ w[1:].T
        w[0] -= eta * (yhat - y).sum() / n
        w[1:] -= eta * (x.T @ (yhat - y)) / n
        costs[i] = cost(y, yhat)

    return w, costs
