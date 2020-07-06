#!/usr/bin/env python

"""
Author: João Ribeiro
"""

import argparse

import numpy as np

from model import FeedForwardNetwork
from utils import load_ocr_dataset, plot


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Hyperparams
    parser.add_argument('-epochs', default=20, type=int, help="Number of training epochs.")
    parser.add_argument('-num_layers', default=2, type=int, help="Number of hidden layers.")
    parser.add_argument('-hidden_size', default=64, type=int, help="Number of units per hidden layer.")
    parser.add_argument('-activation', default="relu", type=str, help="Activation function for the hidden layers.")
    parser.add_argument('-learning_rate', default=0.1, type=float, help="Learning rate for SGD optimizer.")
    parser.add_argument('-l2_penalty', default=0.0, type=float, help="L2 penalty for SGD optimizer.")
    parser.add_argument('-batch_size', default=32, type=int, help="Number of datapoints per SGD step.")

    # Misc.
    parser.add_argument('-data', default='ocr_dataset/letter.data', help="Path to letter.data OCR dataset.")
    parser.add_argument('-save_plot', action="store_true", help="Whether or not to save the generated accuracies plot.")

    opt = parser.parse_args()

    # ############ #
    # Load Dataset #
    # ############ #

    print("Loading OCR Dataset", end="", flush=True)

    data = load_ocr_dataset(opt.data)
    X_train, y_train = data["train"]
    X_val, y_val = data["dev"]
    X_test, y_test = data["test"]
    num_features = X_train.shape[1]
    num_classes = np.unique(y_train).size

    print(" [Done]", flush=True)

    # ########### #
    # Setup Model #
    # ########### #

    print("Deploying model", end="", flush=True)
    model = FeedForwardNetwork(
        num_features, num_classes,
        opt.num_layers, opt.hidden_size, opt.activation,
        opt.learning_rate, opt.l2_penalty, opt.batch_size
    )
    print(" [Done]", flush=True)

    # ################ #
    # Train & Evaluate #
    # ################ #

    print("Training model", flush=True)
    validation_accuracies, final_test_accuracy = model.fit(X_train, y_train, X_val, y_val, X_test, y_test, opt.epochs)

    # #### #
    # Plot #
    # #### #

    print("Plotting", end="", flush=True)
    plot(opt.epochs, validation_accuracies, opt.save_plot)
    print(" [Done]\nGoodbye.", flush=True)
