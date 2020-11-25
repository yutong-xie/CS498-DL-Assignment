"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are the scores for
    each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

        self.outputs = {}
        self.gradients = {}

        # variable for adam optimizer
        self.m = None
        self.v = None
        self.iter = 0


    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        return X.dot(W)+b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return np.maximum(X, 0)

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # implement stable softmax function
        X -= np.max(X, axis = 1, keepdims = True)
        softmax_x =  np.exp(X) / np.sum(np.exp(X), axis = 1, keepdims = True)
        return softmax_x

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.

        # Applied FC layer and relu unit
        input = X
        self.outputs[0] = X
        for i in range(1, self.num_layers):
            self.outputs[i] = input
            w = self.params["W" + str(i)]
            b = self.params["b"+ str(i)]
            y = self.linear(w, input, b)
            y = self.relu(y)
            self.outputs[i] = y
            input = y

        # Applied softmax and output the results
        w = self.params["W" + str(self.num_layers)]
        b = self.params["b"+ str(self.num_layers)]
        y = self.linear(w, input, b)
        self.outputs[self.num_layers] = y

        return self.softmax(y)

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0, beta1: float = 0.9, beta2: float = 0.999, mode: str = "SGD"
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        loss = 0.0
        # TODO: implement me. You'll want to store the gradient of each layer
        # in self.gradients if you want to be able to debug your gradients
        # later. You can use the same keys as self.params. You can add
        # functions like self.linear_grad, self.relu_grad, and
        # self.softmax_grad if it helps organize your code.
        N, D = X.shape

        score = self.outputs[self.num_layers]       # N x C
        score = self.softmax(score)
        # Calculate prediction loss
        loss = -np.sum(np.log(score[np.arange(N), y]))
        loss /= N
        # Calculate L2 regularization for weights of each layer
        for i in range(1, self.num_layers + 1):
            loss += 1 / N * 1 / 2 * reg * np.sum(self.params["W" + str(i)]**2)

        # calculate softmax loss gradient
        dl_dh3 = score.copy()                  # N x C
        dl_dh3[np.arange(N), y] -= 1
        dl_dh3 /= N

        dh3_dw = self.outputs[self.num_layers-1] # N x H
        # calculate linear gradient
        grad_w = np.dot(dh3_dw.T, dl_dh3)        # H x C
        grad_b = np.sum(dl_dh3, axis = 0)   # C,
        self.gradients["W" + str(self.num_layers)] = grad_w
        self.gradients["b" + str(self.num_layers)] = grad_b

        dl_dh = dl_dh3
        for i in range(self.num_layers - 1, 0, -1):

            dh3_dh2 = self.params["W" + str(i+1)]    # H x C
            dl_dh2 = np.dot(dl_dh, dh3_dh2.T)     # N x H
            dh2_dr = self.outputs[i]    # N x H
            # calcualte ReLU gradient
            dh2_dr = (dh2_dr > 0).astype(int)
            dl_dr = dl_dh2 * dh2_dr

            dr_dw2 = self.outputs[i-1]
            grad_w = np.dot(dr_dw2.T, dl_dr) + reg * self.params["W" + str(i)]
            grad_b = np.sum(dl_dr, axis = 0)
            self.gradients["W" + str(i)] = grad_w
            self.gradients["b" + str(i)] = grad_b
            dl_dh = dl_dr

        # SGD optimizer update
        if mode.lower() == "sgd":
            for key in self.params.keys():
                self.params[key] -= lr * self.gradients[key]

        elif mode.lower() == "adam":
            # Adam optimizer update
            if self.m == None:
                self.m, self.v = {}, {}
                for key, val in self.params.items():
                    self.m[key] = np.zeros_like(val)
                    self.v[key] = np.zeros_like(val)

            self.iter += 1
            lr_t = lr * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)

            for key in self.params.keys():
                self.m[key] += (1 - beta1) * (self.gradients[key] - self.m[key])
                self.v[key] += (1 - beta2) * (self.gradients[key]**2 - self.v[key])
                self.params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-8)

        return loss
