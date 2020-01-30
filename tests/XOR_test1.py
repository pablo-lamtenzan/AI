
# this is the first AI tester of this lib, it have to be in /libAI dir to work

"""
In this first test the AI will solve a simple test: the XOR function
"""

import numpy as np

from Network import Network
from Fully_connected_layer import FCLayer
from Activation_layer import ActivationLayer
from Activation_fct import tanh, tanh_p
from Loss_fcts import mean_square_error, mean_square_error_p

# tranig data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_p))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_p))
""" 
    2X -> 3 -> 1 -> Y
"""

# train
net.use(mean_square_error, mean_square_error_p)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)