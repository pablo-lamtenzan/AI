
# Wrappers for primitive Neural Net (NN) Operations.

import numpy as np

class Neural_net :
    __init__(self, features, name) :
        self.features = features
        self.name = name
    
    def relu(self) :
        return (np.maximum(0, self.features)) # relu(x) = 0 if x < 0 else relu(x) = x

    def sigmoid(self) :
        return (1 / (1 + np.exp(-self.features))) # f(x) = (1 + exp(-x))
    
    def sofmax(self) :
        expo = np.exp(self.features)
        expo_sum = np.expo(expo)
        return (expo / expo_sum) # f(x) = exp(x) / sum(exp(x))

    

    
