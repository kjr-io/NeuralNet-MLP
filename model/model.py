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

# Grabbing /input System Path for Imports
sys.path.append(str(f'{os.getcwd()}/input'))
from binaryop_data import *

# Grabbing /utils System Path for Imports
sys.path.append(str(f'{os.getcwd()}/utils'))
from construct_df import *

# Grabbing xor, or, and Data from Input
xor_x_train, xor_y_train = xor_data()
or_x_train, or_y_train = or_data()
and_x_train, and_y_train = and_data()


# Creating the Network
def create_network(x_train, y_train, lr):
    net = Network()
    net.add(FCLayer(2, 3))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(3, 1))
    net.add(ActivationLayer(tanh, tanh_prime))

    net.use(mse, mse_prime)
    net.fit(x_train, y_train, epochs = 10000, learning_rate = lr)

    return net.predict(x_train)

if __name__ == '__main__':

    # NN with Learning Rate of 0.01
    print('-------- XOR TABLE --------')
    preds = create_network(xor_x_train, xor_y_train, 0.01)
    df_1 = construct_xor_df()
    df_1['y1'] = pd.DataFrame(preds)
    print(df_1, '\n')

    print('-------- OR TABLE --------')
    preds = create_network(or_x_train, or_y_train, 0.01)
    df_2 = construct_or_df()
    df_2['y1'] = pd.DataFrame(preds)
    print(df_2, '\n')
    
    print('-------- AND TABLE --------')
    preds = create_network(and_x_train, and_y_train, 0.01)
    df_3 = construct_and_df()
    df_3['y1'] = pd.DataFrame(preds)
    print(df_3, '\n')
