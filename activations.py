import numpy as np


def sigmoid(z, grad=False):
    return 1 / (1 + np.exp(-z)) if not grad else sigmoid(z) * (1 - sigmoid(z))


def relu(z, grad=False):
    if not grad:
        return np.maximum(z, 0)
    else:
        dz = np.zeros_like(z)
        for i in range(z.shape[1]):
            dz[:, i][z[:, i] > 0] = 1
        return dz


def softmax(z):
    exp = np.exp(z)
    return exp / exp.sum(axis=0)


registry = {
    "relu": relu,
    "sigmoid": sigmoid
}
