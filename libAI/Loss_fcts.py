
# loss fcts nad their derivative

import numpy as np

# y_true is the actual value
# y_pred is the prediction 

def mean_square_error(y_true, y_pred) :
    return np.mean(np.power(y_true - y_pred, 2)) # mean -> add all and div by nb of elems 

""" here we are taking the distance between the value we must have and the 
    value than we had (y_true - y_pred) , we penalize further values doing the
    pow """ 

def mean_square_error_p(y_true, y_pred) :
    return 2 * (y_pred - y_true) / y_true.size # mse primitive

