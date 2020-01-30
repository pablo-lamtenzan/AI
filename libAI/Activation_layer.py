
# Activation class

from Layer import Layer

# inherit from class Layer
class ActivationLayer(Layer) :
    def __init__(self, activation, activation_p) :
        self.activation = activation
        self.activation_p = activation_p # activation prime

    # return the activated input
    def forward_propagation(self, input_data) :
        self.input = input_data
        self.output = self.activation(self.input) # f_act(sum(X * W) + b)
        return self.output

    # returns input_error = dE / dX for a given output error = dE / dY
    # learning rete not used (not a learnable parameter)
    def backward_propagation(self, output_error, learning_rate) :
        return self.activation_p(self.input) * output_error 