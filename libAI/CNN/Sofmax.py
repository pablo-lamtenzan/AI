
# standar fully-connected layer using softmax
# could add more thi scode to lib format

import numpy as np

class Softmax() :
    def __init__(self, input_len, nodes) :
        # division is for recrement variance
        self.weights = np.random.rand(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    # input can be an array of any dim
    # returns 1d np array contaning the probabilities values
    def forward(self, X) :
        self.last_input_shape = X.shape # save shape

        # flatten collaps a n dim array into a n = 1 dim array
        X = X.flatten()
        self.last_input = X # save flatted input
        input_len, nodes = self.weights.shape

        totals = self.biases + np.dot(X, self.weights) 
        self.last_totals = totals # save totals

        """ can replace this softmax to class softmax in the lib
            """
        exp = np.exp(totals)
        return exp / np.sum(exp, axis = 0)

    # returns the loss gradient for this layer output
    def backward(self, dE_dY, learning_rate = 0.05) :

        # 1 elem of dE_dY will be nonzero
        for i, grad in enumerate(dE_dY) :
            if grad == 0 :
                continue
        
            # e ^ totals
            t_exp = np.exp(self.last_totals)
            s = np.sum(t_exp)

            # Grad calc , optimization

            # grad of Y[i] against totals
            dY_dTotal = -t_exp[i] * t_exp / (s ** 2)
            dY_dTotal[i] = t_exp[i] * (s - t_exp[i] / s ** 2)

            # grad totals against weight, bias and x
            dTotal_dW = self.last_input
            dTotal_dB = 1
            dTotal_dX = self.weights

            # grad of loss against totals
            dY_dTotals = grad * dY_dTotal

            # grad of loss against weight, bias, x
            dY_dW = dTotal_dW[np.newaxis].T @ dY_dTotal[np.newaxis] #np.newaxis used to increase dims
            dY_dB = dY_dTotal * dTotal_dB
            dY_dX = dTotal_dX @ dY_dTotal

            # update weight and bias
            self.weights -= learning_rate * dY_dW
            self.bias -= learning_rate * dY_dB

            # restore dims
            return dY_dX.reshape(self.last_input_shape)







