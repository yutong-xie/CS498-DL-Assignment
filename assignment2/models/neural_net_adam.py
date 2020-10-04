"""Neural network model using Adam optimizer."""

from typing import Sequence

import numpy as np


class NeuralNetwork_Adam:

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):

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

        return X.dot(W)+b

    def relu(self, X: np.ndarray) -> np.ndarray:

        return np.maximum(X, 0)

    def softmax(self, X: np.ndarray) -> np.ndarray:

        # implement stable softmax function
        X -= np.max(X, axis = 1, keepdims = True)
        softmax_x =  np.exp(X) / np.sum(np.exp(X), axis = 1, keepdims = True)
        return softmax_x

    def forward(self, X: np.ndarray) -> np.ndarray:

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
        self, X: np.ndarray, y: np.ndarray, lr: float, reg:float, beta1: float, beta2: float
    ) -> float:

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
            self.params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


        return loss
