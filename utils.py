from itertools import count
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_ocr_dataset(path, dev_fold=8, test_fold=9):

    label_counter = count()
    labels = defaultdict(lambda: next(label_counter))
    X, y, fold = [], [], []

    with open(path) as f:
        for line in f:
            tokens = line.split()
            pixel_value = [int(t) for t in tokens[6:]]
            letter_class = labels[tokens[1]]
            fold.append(int(tokens[5]))
            X.append(pixel_value)
            y.append(letter_class)

    X, y = np.array(X, dtype='int8'), np.array(y, dtype='int8')

    fold = np.array(fold, dtype='int8')

    train_idx = (fold != dev_fold) & (fold != test_fold)
    X_train, y_train = X[train_idx], y[train_idx]

    val_idx = fold == dev_fold
    X_val, y_val = X[val_idx], y[val_idx]

    test_idx = fold == test_fold
    X_test, y_test = X[test_idx], y[test_idx]

    return {
        "train": (X_train, y_train),
        "dev": (X_val, y_val),
        "test": (X_test, y_test)
    }


def sample_batch(X, y, batch_size):

    M = len(X)
    B = batch_size

    min_batch_indices = np.random.choice(M, B)

    X_batch = np.array([X[i] for i in min_batch_indices])
    y_batch = np.array([y[i] for i in min_batch_indices])

    return X_batch, y_batch


def one_hot_encoding(y, num_classes):
    y_one_hot = np.zeros((num_classes, y.shape[0]))
    for i, value in enumerate(y):
        y_one_hot[value, i] = 1
    return y_one_hot


def plot(epochs, validation_accuracies, save=False):
    plt.title(f"Training {epochs} epochs on the OCR dataset")
    epochs = np.arange(1, epochs + 1)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Set Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, validation_accuracies, color="orange")
    plt.show()
    if save:
        plt.savefig("plot.png")
    plt.close()
