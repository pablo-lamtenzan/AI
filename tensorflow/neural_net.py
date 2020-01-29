
# Wrappers for primitive Neural Net (NN) Operations.

import numpy as np

class Neural_net :
    __init__(self, features, name) :
        self.features = features
        self.name = name
    
    def relu(self, features) :
        return (np.maximum(0, features)) # relu(x) = 0 if x < 0 else relu(x) = x

    def sigmoid(self, features) :
        return (1 / (1 + np.exp(-features))) # f(x) = 1 / (1 + exp(-x))
    
    def sofmax(self, features) :
        expo = np.exp(features)
        expo_sum = np.sum(expo)
        return (expo / expo_sum) # f(x) = exp(x) / sum(exp(x))

    

    
