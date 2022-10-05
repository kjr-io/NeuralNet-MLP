import os
import sys
import matplotlib.pyplot as plt

sys.path.append(str(f'{os.getcwd()}/utils'))
from construct_plot import *

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        result_clean = []

        for i in range(samples):
            output = input_data[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)

            result.append(output)

        for item_ in result:
            item = str(item_)

            sliced = item[2:]
            sliced = sliced[:-2]

            result_clean.append(sliced)

        return result_clean

    def fit(self, x_train, y_train, epochs, learning_rate, plot_details):
        errStore = []
        for _ in range(epochs):
            err = 0
    
            for j in range(len(x_train)):
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            errStore.append(err)
        

        construct_plot(epochs, errStore, learning_rate, plot_details)
        '''
        plt.plot(range(epochs), errStore, "-b", label=learning_rate)
        plt.legend(loc='upper right')
        plt.savefig('./output/test.png')
        '''

        print(f'Error: {err}')
