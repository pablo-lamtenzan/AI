
# some cool training classes

# https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f

import numpy as np

Class train :
    __init__(self, features, learning_rate, max_step) : #must chage that
        self.features = features
        self.learning_rate = learning_rate
        self.max_step = max_step
    """
        this is out tensorflow lib is just an explenated example:
        with the cost fct we evalueate the predictions and with grad fct we minimize error
    """
    def Cost_fct(theta, x, y) :
        _len = len(y)
        predict = x.dot(theta) # predicted values 
        cost = (1 / 2 * m) * np.sum(np.square(predict - y)) # tipical mean quad error eq | 2 * m | m = ?
        return (cost)

    def GradientDescent(theta, x, y, learnig_rate, it) : #this is a example for linear regresion
        """
        x = bias
        y = vect
        theta start random
        """
        _len = len(y)
        cost_hist = np.zeros(it)
        theta_hist = np.zeros(it, 2)
        for i in range(it) :
            predict = np.dot(x, theta)
            theta = theta - (1 / _len) * learning_rate * (x.t.dot((predict - y))) # evaluation then minimize error
            theta_hist[i,:] = theta.T
            cost_hist[i] = Cost_fct(theta, x, y) # takes values of how theta dicrement (if we minize error succesfully)
        return (theta, cost_hist, theta_hist)

    def 
