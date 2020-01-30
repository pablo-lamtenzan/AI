
# activation fct and derivative

import numpy as np

def tanh(x) : # hiperbolic tangent
    return np.tanh(x)

def tanh_p(x) : # hiperbolic tangent primitive
    return 1 - np.tanh(x) ** 2

def relu(x) :
    return np.maximum(0, x) # relu(x) = 0 if x < 0 else relu(x) = x

def relu_p(x) :
    return 1 if x >= 0 else 0 # relu primitive

def sigmoid(x) :
    return 1 / 1 + np.exp(-x) # sigmoid(x) = 1 / (1 + exp(-x))
"""
def sigmoid_p(x)
    return sigmoid(x) * (1 - sigmoid(x)) # sigmoid primitive
"""
def softmax(x) : 
        expo = np.exp(x)
        expo_sum = np.sum(expo)
        return (expo / expo_sum) # softmax(x) = exp(x) / sum(exp(x))

""" have to add softmax_p but there exist a lot of act fct more :) """



