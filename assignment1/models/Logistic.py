"""Logistic regression model."""

import numpy as np
from tqdm import tqdm

class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me

        return 1/(1 + np.exp(-1.0 * z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = X_train.shape
        # Initialize weights and bias
        self.w = np.ones(D+1)
        # Add a dummy feature for each training sample
        X_train = np.insert(X_train, D, values = 1, axis = 1)
        # tqdm is used to indicate the progress
        for _ in tqdm(range(self.epochs)):
            for i in range(N):
                y_gt = 1 if y_train[i] == 1 else -1
                y_pred = np.dot(X_train[i], self.w)
                self.w += self.lr * (self.sigmoid(-y_gt * y_pred) * y_gt * X_train[i])

        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = np.insert(X_test, X_test.shape[1], values = 1, axis = 1)
        y_pred = np.dot(X_test, self.w) > 0

        return y_pred
