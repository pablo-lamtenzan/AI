
# Error functions

import numpy as np

# y_true are the true values
# y_pred are the prediction of this values
# n as the nb of elems
# MSE = (sum(square(y_pred - y_true))) / n
# if u don t already know this is a cost function :)

def mean_squared_error(y_true, y_pred) :
    return (np.square(np.subtract(y_true, y_pred)).mean()) # .mean() = sum of elems allog the axis and vidided by their nb


