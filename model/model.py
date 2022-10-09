import sys
import os
import numpy as np

# Grabbing /src System Path for Imports
sys.path.append(str(f'{os.getcwd()}/src'))

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_layer import tanh, tanh_prime
from loss_func import mse, mse_prime

# Creating the Network
def create_network(x_train, y_train, lr, plot_details):
    net = Network()
    net.add(FCLayer(2, 3))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(3, 1))
    net.add(ActivationLayer(tanh, tanh_prime))

    net.use(mse, mse_prime)
    net.fit(x_train, y_train, epochs = 5000, learning_rate = lr, plot_details = plot_details)

    return net.predict(x_train)

