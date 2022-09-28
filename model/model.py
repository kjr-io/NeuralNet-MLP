import sys
import os
import numpy as np

sys.path.append('/Users/kyleryan/Documents/GitHub/NeuralNet-MLP/src')

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_layer import tanh, tanh_prime
from loss_func import mse, mse_prime

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Creating the Network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs = 10000, learning_rate = 0.1)

out = net.predict(x_train)
print(out)