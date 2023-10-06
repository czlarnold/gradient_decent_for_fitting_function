# -*- coding: utf-8 -*-
"""
# File        : gradient_decent.py
# Time        : 2023/10/05  05:52
# Author      : CAI Zl
# Software    : PyCharm
# Version     : python 3.9
# Description : 
"""

import numpy as np
import matplotlib.pyplot as plt


class Optimizer(object):

    def __init__(self, lr: float, num_params: int):
        self.lr = lr
        self.num_params = num_params
        self.optim_count = 0

    def optim(self, params: np.array, gradient: np.array) -> np.array:
        pass

    @property
    def run_step(self):
        return self.optim_count


class SGD(Optimizer):

    def __init__(self, lr: float, num_params: int):
        super().__init__(lr, num_params)

    def optim(self, params: np.array, gradient: np.array) -> np.array:
        self.optim_count += 1

        params -= self.lr * gradient
        return params


class Momentum(Optimizer):

    def __init__(self, lr: float, num_params: int, beta=0.9):
        super().__init__(lr, num_params)
        self.beta = beta
        self.v = np.array([0]*self.num_params)

    def optim(self, params: np.array, gradient: np.array) -> np.array:
        self.optim_count += 1

        self.v = self.beta * self.v + (1-self.beta) * gradient
        params -= self.lr * self.v
        return params


class Adam(Optimizer):

    def __init__(self, lr, num_params,
                 beta1=0.9, beta2=0.999, epsilon=1e-8,
                 if_corrected=False):
        super().__init__(lr, num_params)

        self.beta1 = beta1
        self.v = np.array([0] * self.num_params)
        self.beta2 = beta2
        self.S = np.array([0] * self.num_params)

        self.if_corrected = if_corrected

        self.epsilon = epsilon

    def optim(self, params: np.array, gradient: np.array) -> np.array:
        self.optim_count += 1

        self.v = self.beta1 * self.v + (1 - self.beta1) * gradient
        self.S = self.beta2 * self.S + (1 - self.beta2) * np.square(gradient)

        if self.if_corrected:
            self.v = self.v / (1 - np.power(self.beta1, self.optim_count))
            self.S = self.S / (1 - np.power(self.beta1, self.optim_count))

        params -= self.lr * self.v / (np.sqrt(self.S) + self.epsilon)
        return params

if __name__ == '__main__':
    lr = 0.1
    num_params = 3

    params = np.random.randn(num_params) * 100
    gradient = np.random.randn(num_params)

    SGD_p = Momentum_p = Adam_p_1 = Adam_p_2 = params

    print("raw params: {}".format(params))
    print("gradient: {}".format(gradient))

    SGD_Optimizer = SGD(lr, num_params)
    Momentum_Optimizer = Momentum(lr, num_params)
    Adam_Optimizer_1 = Adam(lr, num_params)
    Adam_Optimizer_2 = Adam(lr, num_params, if_corrected=True)

    SGD_list = [SGD_p]
    momentum_list = [Momentum_p]
    Adam_list_1 = [Adam_p_1]
    Adam_list_2 = [Adam_p_2]
    for i in range(100):
        # print("----- {} -----".format(i))
        # print("SGD: {}".format(SGD_Optimizer.SGD(params, gradient)))
        # print("Momentum: {}".format(Momentum_Optimizer.momentum(params, gradient)))
        # print("Adam: {}".format((Adam_Optimizer.Adam(params, gradient))))
        # print()
        SGD_p = SGD_Optimizer.optim(SGD_p, gradient)
        SGD_list.append(SGD_p.tolist())

        Momentum_p = Momentum_Optimizer.optim(Momentum_p, gradient)
        momentum_list.append(Momentum_p.tolist())

        Adam_p_1 = Adam_Optimizer_1.optim(Adam_p_1, gradient)
        Adam_list_1.append(Adam_p_1.tolist())

        Adam_p_2 = Adam_Optimizer_2.optim(Adam_p_2, gradient)
        Adam_list_2.append(Adam_p_2.tolist())

        # gradient = np.random.randn(num_params)

    SGD_array = np.array(SGD_list)
    momentum_array = np.array(momentum_list)
    Adam_array_1 = np.array(Adam_list_1)
    Adam_array_2 = np.array(Adam_list_2)

    fig, ax = plt.subplots()
    ax.set_title("Param 1 with different optimizer")
    ax.plot(SGD_array[:, 0], label="SGD")
    ax.plot(momentum_array[:, 0], label="momentum")
    ax.plot(Adam_array_1[:, 0], label="Adam without correcter")
    ax.plot(Adam_array_2[:, 0], label="Adam with correcter")
    ax.grid()
    ax.legend()
    plt.show()

