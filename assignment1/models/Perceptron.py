"""Perceptron model."""

import numpy as np
from tqdm import tqdm

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            d_attr: the number of dimensions of X_train
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement
        N, D = X_train.shape
        self.w = np.random.rand(self.n_class, D)

        for _ in tqdm(range(self.epochs)):
            for i in range(N): # iterate over each sample
                y_i = y_train[i]
                x_i = X_train[i]
                y_pred = self.w @ x_i
                y_pred_class = np.argmax(y_pred)
                if y_pred_class != y_i:
                  for c in range(self.n_class): #iterate over each classes c
                    if self.w[c] @ x_i > self.w[y_i] @ x_i:
                      self.w[c] = self.w[c] - self.lr * x_i
                      self.w[y_i] = self.w[y_i] + self.lr * x_i

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
        y_pred = self.w @ np.transpose(X_test)
        y_output = np.argmax(y_pred, axis = 0)

        return y_output
