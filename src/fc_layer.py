from base_layer import Layer
import numpy as np

# Constructing the Fully Connected Layer Class
class FCLayer(Layer):
    def __init__(self, input_length, output_length):
        '''
        Initializing the Weights and Biases
        '''
        self.weights = np.random.rand(input_length, output_length) - 0.5
        self.biases = np.zeros(output_length) + 0.1

    def forward_propagation(self, input):
        '''
        Forward Propagation Function
        '''
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        '''
        Backward Propagation Function
        Assigning Weights and Biases
        '''
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.biases = self.biases - (learning_rate * output_error)

        return input_error