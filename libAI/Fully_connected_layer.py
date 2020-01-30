
# FCLayer obj

from Layer import Layer
import numpy as np

# inherit from class Layer
# input_size is nb of imput neurons
# output_size is nb of output neurons
class FCLayer(Layer) :
    def __init__(self, input_size, output_size) :
        self.weight = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        """ weith and bias is more or less "random" in the init but the value
            will be corrected during the trining """

    def forward_propagation(self, input_data) :
        self.input = input_data
        self.output = np.dot(self.input, self.weight) + self.bias # y = b + sum(X * W)
        return self.output

    # computes dE / dW, dE / dB for a given dE / dY
    # return input_error = dE / dX
    def backward_propagation(self, output_error, learnig_rate) :
        input_error = np.dot(output_error, self.weight.T) # .T means the transpose of the tensor
        weight_error = np.dot(self.input.T , output_error) # dB = output_error
        # update weight and bias 
        self.weight -= learnig_rate * weight_error
        self.bias -= learnig_rate * output_error
        return input_error

