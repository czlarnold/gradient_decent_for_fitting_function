# -*- coding: utf-8 -*-
"""
# File        : functions.py
# Time        : 2023/10/05  05:51
# Author      : CAI Zl
# Software    : PyCharm
# Version     : python 3.9
# Description : 
"""

import numpy as np
import matplotlib.pyplot as plt


class ModuFunc(object):
    """ Modul Function

        Func predict_y will return the result of before parameter
        renew.

        Func __predict_y will calculate the predict result.

        Func __loss will calculate the difference of obverse result
        and predict result.

        Func loss will calculate the MSE of model as above:
            L = 1/2 * sum(loss^2) / length

        Func gradient will return gradient of working parameters. But
        the gradient must be defined.
    """

    def __init__(self, data: np.array):
        """

        Parameter x is obverse state, while y is obverse result.

        :param data: x: data[:-1], y: data[-1].
        """
        self.x = data[:-1]
        self.y = data[-1]
        self.length = len(self.y)

    def predict_y(self) -> np.array:
        pass

    def __predict_y(self) -> np.array:
        pass

    def __loss(self) -> np.array:
        pass

    def loss(self) -> np.array:
        pass

    def gradient(self) -> np.array:
        pass


class FunctionTemplate(ModuFunc):
    """ Function Template

        On this template, the function is shown as above:
            f(x) = a / (b + exp(-c*x+d))
        Number of working parameters is 4, while Number of support
        parameter is 0.
    """
    working_params_num = 4  # Using on training

    def __init__(self, data: np.array,
                 working_params: np.array,
                 support_params: np.array):
        """

        :param data: a dataset with x and y.
        :param working_params: the parameters need to fitting.
        :param support_params: the static parameters.
        """
        super().__init__(data)

        self.a = working_params[0]
        self.b = working_params[1]
        self.c = working_params[2]
        self.d = working_params[3]

        self.__u = - self.c * self.x + self.d
        self.__v = np.exp(self.__u)
        self.__w = 1 + self.b * self.__v

        self.__pred_y = self.__predict_y()
        self.__ll = self.__loss()

    def predict_y(self) -> np.array:
        return self.__pred_y

    def __predict_y(self) -> np.array:
        return self.a / self.__w

    def __loss(self) -> np.array:
        return self.y - self.__pred_y

    def loss(self) -> np.array:
        return np.sum(np.square(self.__ll)) / (2 * self.length)

    def gradient(self) -> np.array:
        r = [self.dL_da(), self.dL_db(), self.dL_dc(), self.dL_dd()]
        return np.array(r)

    def dL_da(self):
        return np.sum(self.__ll * (-1 / self.__w)) / self.length

    def dL_db(self):
        r = self.a / np.square(self.__w)
        return np.sum(self.__ll * r * self.__v) / self.length

    def dL_dc(self):
        r = self.a / np.square(self.__w)
        s = - self.x
        return np.sum(self.__ll * r * self.__v * s) / self.length

    def dL_dd(self):
        r = self.a / np.square(self.__w)
        return np.sum(self.__ll * r * self.__v) / self.length


def noise(noise_data: np.array, noise_coef=0.1):
    return noise_data + np.random.randn(len(noise_data)) * noise_coef


if __name__ == '__main__':
    def test_func(x, a, b, c, d):
        u = -c * x + d
        v = b * np.exp(u)
        w = 1 + v
        return a / w

    data_num = 1000
    x = np.linspace(50, 450, data_num)
    func_working = np.array([30, 0.1, 0.008, 1])
    func_support = np.array([])
    y = test_func(x,
                  a=func_working[0],
                  b=func_working[1],
                  c=func_working[2],
                  d=func_working[3])
    xx = noise(x, 0.01)
    yy = noise(y, 0.03)

    data_set = np.array([xx, yy])

    working_params = np.array([29.1, 0.12, 0.0081, 1.1])
    support_params = np.array([])
    solver = FunctionTemplate(data_set, working_params, support_params)
    pred_y = solver.predict_y()
    loss = solver.loss()
    gradient = solver.gradient()
    print(f"loss: {loss}")
    print(f"gradient: {gradient}")

    fig, ax = plt.subplots()
    ax.set_title("real, target & predict")
    ax.plot(x, y, label="target result")
    ax.plot(xx, yy, "*", label="real obverse")
    ax.plot(xx, pred_y[0], label="fitting result")
    ax.grid()
    ax.legend()
    plt.show()
