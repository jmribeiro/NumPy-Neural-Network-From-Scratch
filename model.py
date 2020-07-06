import math

import numpy as np
from activations import registry, softmax
from utils import sample_batch, one_hot_encoding


class FeedForwardNetwork:

    def __init__(self, num_features: int, num_classes: int,
                 num_layers: int, hidden_size: int, activation: str,
                 learning_rate: float, l2_penalty: float, batch_size: int):

        self.layers = []

        last_units = num_features
        for _ in range(num_layers):
            self.layers.append((Linear(last_units, hidden_size), registry[activation]))
            last_units = hidden_size
        self.layers.append((Linear(last_units, num_classes), softmax))

        self.Z_cache = []
        self.H_cache = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_features = num_features
        self.num_classes = num_classes
        self.l2_penalty = l2_penalty

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs, verbose=True):

        num_datapoints = X_train.shape[0]

        validation_accuracies = []
        for e in range(epochs):

            if verbose:
                print(f"\tEpoch {e+1}/{epochs}", end="", flush=True)

            for _ in range(0, num_datapoints, self.batch_size):
                X_batch, y_batch = sample_batch(X_train, y_train, self.batch_size)
                self.sgd_step(X_batch, y_batch)

            validation_accuracies.append(self.accuracy(X_val, y_val))

            if verbose:
                print(f" -> Val. acc.: {round(validation_accuracies[-1], 5)}", flush=True)

        final_test_accuracy = self.accuracy(X_test, y_test)

        if verbose:
            print(f"Training complete!\n\tFinal test acc.: {round(final_test_accuracy, 5)}", flush=True)

        return validation_accuracies, final_test_accuracy

    def sgd_step(self, X, y):

        # Forward propagation
        y_hat = self.forward(X)

        # Compute gradient
        dZ = categorical_cross_entropy(one_hot_encoding(y, self.num_classes), y_hat, grad=True)

        # Backward propagation
        self.backward(dZ, X)

    def forward(self, X):

        self.Z_cache.clear()
        self.H_cache.clear()

        H = X.T

        for layer, activation in self.layers:
            Z = layer.forward(H)
            H = activation(Z)
            self.Z_cache.append(Z)
            self.H_cache.append(H)

        y_hat = H
        return y_hat

    def backward(self, dZ, X):

        for l in reversed(range(len(self.layers))):
            first_layer = l == 0

            layer, _ = self.layers[l]
            H_prev = X.T if first_layer else self.H_cache[l - 1]
            dW, dB = layer.grad(dZ, H_prev)

            if not first_layer:
                g_prev = self.layers[l - 1][1]
                Z_prev = self.Z_cache[l - 1]
                dZ = layer.backward(dZ, Z_prev, g_prev)

            layer.W -= self.learning_rate * (self.l2_penalty * layer.W + dW)
            layer.b -= self.learning_rate * dB

    def predict(self, X):
        y_hat = self.forward(X)
        labels = y_hat.argmax(axis=0)
        return labels

    def accuracy(self, X, y):
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Linear:

    """ PyTorch Style Linear/Affine/Dense/Hidden Layer """

    def __init__(self, num_inputs, num_outputs):
        # Default glorot initialization
        t = math.sqrt(6) / math.sqrt(num_outputs + num_inputs)
        self.W = np.random.uniform(-t, t, (num_outputs, num_inputs))
        self.b = np.zeros((num_outputs, 1))

    def forward(self, H):
        return np.dot(self.W, H) + self.b

    def backward(self, dZ, Z_prev, g):
        dH_prev = np.dot(self.W.T, dZ)
        dZ_prev = dH_prev * g(Z_prev, grad=True)
        return dZ_prev

    def grad(self, dZ, H_prev):
        B = dZ.shape[1]
        dW = (1.0 / B) * np.dot(dZ, H_prev.T)
        db = (1.0 / B) * np.sum(dZ, axis=1, keepdims=True)
        return dW, db


def categorical_cross_entropy(y_one_hot, y_hat, grad=False):

    if not grad:

        batch_size = y_one_hot.shape[1]

        # Compute individual losses
        losses = np.multiply(y_one_hot, np.log(y_hat))

        # Sum all
        loss = np.sum(losses)

        # Average over batch
        average_loss = -(1.0 / batch_size) * loss

        return average_loss

    else:

        # Handcoded gradient
        return y_hat - y_one_hot