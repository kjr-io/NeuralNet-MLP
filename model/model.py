import sys
import os
import numpy as np

sys.path.append(str(f'{os.getcwd()}/src'))

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_layer import tanh, tanh_prime
from loss_func import mse, mse_prime

sys.path.append(str(f'{os.getcwd()}/input'))
from binaryop_data import *

x_train, y_train = xor_data()


# Creating the Network
def create_network(x_train, y_train):
    net = Network()
    net.add(FCLayer(2, 3))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(3, 1))
    net.add(ActivationLayer(tanh, tanh_prime))

    net.use(mse, mse_prime)
    net.fit(x_train, y_train, epochs = 10000, learning_rate = 0.01)

    out = net.predict(x_train)
    print(out)

if __name__ == '__main__':
    create_network(x_train, y_train)