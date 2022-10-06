import os
import sys
import matplotlib.pyplot as plt

sys.path.append(str(f'{os.getcwd()}/utils'))
from construct_plot import *

# Constructing the Neural Network Class
class Network:
    def __init__(self):
        ''' 
        Initializing Neural Network
        '''
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        '''
        Adding Layers to Neural Network
        '''
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        '''
        Using Loss & Loss Prime as Loss Functions
        '''
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        '''
        Running Predict Based on Input Data
        '''
        samples = len(input_data)
        result = []
        result_clean = []


        for i in range(samples):
            output = input_data[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)

            result.append(output)

        # Cleaning Up Forward Propagation Output for Readability
        for item_ in result:
            item = str(item_)

            sliced = item[2:]
            sliced = sliced[:-2]

            result_clean.append(sliced)

        return result_clean

    def fit(self, x_train, y_train, epochs, learning_rate, plot_details):
        '''
        Fitting the Model to the Data
        Storing the Error in an Arr to Graph Using Matplotlib
        Constructing Graph
        '''
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
        print(f'Error: {err}')
