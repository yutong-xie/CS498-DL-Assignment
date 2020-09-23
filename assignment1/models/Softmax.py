"""Softmax model."""

import numpy as np
from tqdm import tqdm

class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, batch_size: int, random: bool):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_size = batch_size
        self.random = random

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement m

        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = X_train.shape
        # Initialize weights and bias
        self.w = np.ones((self.n_class, D+1))
        # Add a dummy feature for each training sample
        X_train = np.insert(X_train, D, values = 1, axis = 1)
        for _ in tqdm(range(self.epochs)):
            if self.random:
                randIndex = np.random.choice(range(N), self.batch_size, replace = False)
            else:
                randIndex = range(N)
                
            for i in randIndex:
                y_gt = y_train[i]
                y_pred = np.dot(self.w, X_train[i])
                y_pred -= np.max(y_pred)
                y_pred = np.exp(y_pred)/np.sum(np.exp(y_pred))

                for j in range(self.n_class):
                    if j == y_gt:
                        self.w[j] -= self.lr * (y_pred[j] - 1) * X_train[i]
                    else:
                        self.w[j] -= self.lr * y_pred[j] * X_train[i]

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
        y_pred = np.dot(X_test, self.w.T)
        return np.argmax(y_pred, axis=1)
