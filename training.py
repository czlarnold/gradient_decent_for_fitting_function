# -*- coding: utf-8 -*-
"""
# File        : training.py
# Time        : 2023/10/05  05:53
# Author      : CAI Zl
# Software    : PyCharm
# Version     : python 3.9
# Description : 
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from functions import FunctionTemplate
from gradient_decent import SGD, Momentum, Adam


def training(data_set, FUNC, init_func_params: dict, optim,
             max_iter=10000, lr=0.001, stop_loss=0.001):
    working_params = init_func_params["working_params"]
    working_params_num = init_func_params["working_params_num"]
    support_params = init_func_params["support_params"]

    opt = optim(lr, working_params_num)

    # init running params
    best_loss = 500.00
    best_params = []
    best_iter = 0

    pred_params = working_params

    # running
    for i in range(max_iter):
        solver = FUNC(data_set, working_params, support_params)
        loss = solver.loss()

        if loss < stop_loss:
            best_loss = loss
            best_params = pred_params
            print("--- stop by best {}: loss={:.4f}, bess_loss={:.4f}".format(i + 1, loss, best_loss))
            break

        if i - best_iter > 500 and best_iter != 0:
            print("--- stop by stable {}: loss={:.4f}, bess_loss={:.4f}".format(i+1, loss, best_loss))
            break

        if (i+1) % 1000 == 0:
            print("--- {}: loss={:.4f}, bess_loss={:.4f}".format(i + 1, loss, best_loss))

        if loss < best_loss:
            print("--- better {}: loss={:.4f}, bess_loss={:.4f}".format(i + 1, loss, best_loss))
            best_loss = loss
            best_iter = i
            best_params = pred_params

        gradient = solver.gradient()

        pred_params = opt.optim(pred_params, gradient)

    if len(best_params) == 0:
        best_params = working_params
        best_loss = np.nan

    return best_params, best_loss

def test_func(x: np.array, working_params, support_params=None):
    a = working_params[0]
    b = working_params[1]
    c = working_params[2]
    d = working_params[3]

    u = -c * x + d
    v = b * np.exp(u)
    w = 1 + v
    return a / w


def noise(noise_data: np.array, noise_coef=0.1):
    return noise_data + np.random.randn(len(noise_data)) * noise_coef


if __name__ == '__main__':
    num_of_data = 2000
    x = np.linspace(250, 1250, num_of_data)
    func_params = (30, 100, 0.008, 1)
    y = test_func(x, working_params=func_params)

    xx = noise(x, 0.01)
    yy = noise(y, 0.3)

    working_dict = dict()
    working_dict["target_func"] = test_func
    working_dict["lr"] = 0.0001
    working_dict["beta"] = 0.7
    working_dict["data_percent"] = 0.7
    working_dict["init_params"] = {"a": np.random.uniform(25, 35),
                                   "b": np.random.uniform(95, 105),
                                   "c": np.random.uniform(0, 0.01),
                                   "d": np.random.uniform(0.5, 1.5)}
    working_dict["best_params"] = [0, 0, 0, 0]
    working_dict["fitting_func"] = FunctionTemplate
    working_dict["func_working"] = {
        "working_params": np.array([working_dict["init_params"]["a"],
                                    working_dict["init_params"]["b"],
                                    working_dict["init_params"]["c"],
                                    working_dict["init_params"]["d"]]),
        "support_params": None,
        "working_params_num": working_dict["fitting_func"].working_params_num
    }
    working_dict["optim"] = Adam
    working_dict["re_init_params"] = {
        0: working_dict["init_params"]
    }

    # run parameters
    epochs = 10
    best_epoch_loss = 1000
    best_epoch = 0
    epoch_loss_line = 10
    epochs_loss = []
    training_loss = []
    all_params = []
    max_iter = 10000
    data_label = [i for i in range(num_of_data)]
    data_number = int(working_dict["data_percent"] * num_of_data)

    for epoch in range(epochs):
        data_choice = random.sample(data_label, data_number)
        sample_yy = yy[data_choice]
        sample_xx = xx[data_choice]

        data_set = np.array([sample_xx, sample_yy])

        # epoch_params, gd_loss = gradient_decrease(sample_xx, sample_yy,
        #                                           working_dict["fitting_func"], working_dict["func_working"],
        #                                           max_iter=max_iter, lr=working_dict["lr"])
        epoch_params, gd_loss = training(data_set,
                                         working_dict["fitting_func"], working_dict["func_working"],
                                         working_dict["optim"],
                                         max_iter=max_iter, lr=working_dict["lr"])
        training_loss.append(gd_loss)

        epoch_y = working_dict["target_func"](xx, epoch_params)
        epoch_loss = np.sum(np.square(yy - epoch_y)) / num_of_data
        epochs_loss.append(epoch_loss)

        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch = epoch
            working_dict["best_params"] = epoch_params
        all_params.append(epoch_params)

    new_y = working_dict["target_func"](xx, working_dict["best_params"], working_dict["func_working"]["support_params"])
    print("best epoch: {}, best loss: {:.4f}".format(best_epoch, best_epoch_loss))
    print("best params:", working_dict["best_params"])
    print("random init params:", working_dict["init_params"])

    fig, ax = plt.subplots()
    ax.set_title("real, target & predict")
    ax.plot(x, y, label="target result")
    ax.plot(xx, yy, "*", label="real obverse")
    ax.plot(xx, new_y, label="fitting result")
    ax.grid()
    ax.legend()
    plt.show()

    fig_loss, ax_loss = plt.subplots()
    ax_loss.set_title("epochs loss & training loss")
    ax_loss.plot(epochs_loss, label="epochs loss")
    ax_loss.plot(training_loss, label="training loss")
    ax_loss.grid()
    ax_loss.legend()
    plt.show()

