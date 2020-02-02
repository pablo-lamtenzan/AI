
import numpy as np

""" 
TO DO :

- Merge classes adding params
- Repair segfault linear regresion

- All DO TO writen in this file
- Cross Entropy product in loss
- Add all existing activation function
- Add an independent Genetic Lib
- Add CNN lib
- Add RNN lib
- Add QL and DQL lib
- Add all optimizers
- Add loss ftc
- Implement an kind of .h for only one inclusion
- Reimplement numpy
- Implement data graph visualition lib
- Implement residual neural network

- When will finished a man

- End snake (includes)
- Repair AE
- ...

Did :

- Add Deep Learning lib base
- Add genetic lib


"""


"""----------------------------------------------------  DEPENDECIES  ----------------------------------------"""

# new tensor
class Tensor() :
    def __init__(self, shape) :
        self.data = np.ndarray(shape, np.float32)
        self.grad = np.ndarray(shape, np.float32)

# function abstract class provides an interface for operators
class Function(object) :

    def forward(self) :
        raise NotImplementedError

    def backward(self) :
        raise NotImplementedError

    def getParams(self) :
        return []

# abstrac class provides an interface from optimizers
class Optimizer(object) :
    def __init__(self, params) :
        self.params = params
    
    def step(self) :
        raise NotImplementedError

    def zeroGrad(self) :
        for param in self.params :
            param.grad = 0.0




"""-----------------------------------------------  LAYER  ------------------------------------------------------"""

class Linear(Function) :
    def __init__(self, in_nodes, out_nodes) :
        self.weights = Tensor((in_nodes, out_nodes))
        self.bias = Tensor((1, out_nodes))
        self.type = 'linear'

    # h = sb + sum(x * w)
    def forward(self, X) :
        h = np.dot(X, self.weights.data) + self.bias.data
        self.input = X
        return h

    # use dE / dY for get dE / dW and dE / dB
    def backward(self, dY) :
        self.weights.grad += np.dot(self.input.T, dY)
        self.bias.grad += np.sum(dY, axis = 0, keepdims = True)
        grad_input = np.dot(dY, self.weights.data.T)
        return grad_input
    """ return the partial derivatives with respect to thw input X,
        that willl be passed on to the previous layer """
    
    # data visuslaization 
    def getParams(self) :
        return [self.weights, self.bias]


"""-------------------------------------------------  LOSS  -----------------------------------------------------"""

class SoftmaxWithLoss(Function) :
    def __init__(self) :
        self.type = 'normalization'

    # softmax fonction as normalization fct, loss fct
    def forward(self, x, target) :
        unnormalized_proba = np.exp(x - np.max(x, axis = 1, keepdims = True))
        self.proba = unnormalized_proba / np.sum(unnormalized_proba, axis = 1, keepdims = True)
        self.target = target
        loss = -np.log(self.proba[range(len(target)), target])
        return loss.mean()

    # softmax primitive 
    def backward(self) :
        grad = self.proba
        grad[range(len(self.target)), self.target] -= 1.0
        grad /= len(self.target)
        return grad

# mean cuadratic error
class MSE(Function) :
    def __init__(self) :
        self.type = 'loss'

    # mse error rate
    def forward(self, Y_true, Y_pred) :
        return np.mean(np.power(Y_true - Y_pred, 2))

    # mse primitive
    def backward(self, Y_true, Y_pred) :
        return 2 * (Y_pred - Y_true) / Y_true.size

# mean absolute error
class MAE(Function) :
    def __init__(self) :
        self.type = 'loss'

    # mae error rate
    def forward(self, Y_true, Y_pred) :
        return np.sum(np.absolute(Y_true - Y_pred))

    def backward(self, Y_true, Y_pred) :
        return 0 # TO DO


class CrossEntropy(Function) :
    def __init__(self) :
        self.type = 'loss'

    # cross entropy error rate
    def forward(self, Y_true, Y_pred) :
        if Y_pred == 1 :
            return -np.log(Y_true)
        else :
            return -np.log(1 - Y_true)

    # cross entropy primitive
    def backward(self, Y_true, Y_pred) :
        return 0 # TO DO

# used for clasification
class Hinge(Function) :
    def __init__(self) :
        self.type = 'loss'

    # Hinge error rate
    def forward(self, Y_true, Y_pred) :
        return np.max(0, 1 - Y_true * Y_pred)

    # Hinge derivative
    def backward(self, Y_true, Y_pred) :
        return 0 # TO DO

# use for regresion, less sensitive that mse
class Huber(Function) :
    def __init__(self) :
        self.type = 'loss'

    # huber error rate 
    def forward(self, Y_true, Y_pred, delta = 1) :
        return np.where(np.abs(Y_pred - Y_true) < delta, 0.5 * (Y_pred - Y_true) ** 2,
         delta * (np.abs(Y_pred - Y_true) - 0.5 * delta))

    # huber derivative
    def backward(self, Y_true, Y_pred, delta = 1) :
        return 0 # TO DO

# Kullback - Leibler
class KLDivergence(Function) :
    def __init__(self) :
        self.type = 'loss'
    
    # KLD error rate
    def forward(self, Y_true, Y_pred) :
        return np.sum(Y_true * np.log(Y_true / Y_pred))

    def backward(self, Y_pred, Y_pred) :
        return 0 # TO DO








"""---------------------------------------------  ACTIVATION FUNCTIONS  ----------------------------------------"""

class ReLU(Function) :
    def __init__(self, inplace = True) :
        self.type = 'activation'
        self.inplace = inplace
    
    # ReLU fct as activation
    def forward(self, x) :
        # x = x < 0 ? 0 : x
        if self.inplace :
            x[x < 0] = 0.0
            self.activated = x
        else :
            self.activated = x * (x > 0)

        return (self.activated)

    # ReLU prmitive
    def backward(self, dY) :
        return dY * (self.activated > 0)

class LeakyReLU(Function) : """in the future i could join all relu fct in 1"""
    def __init__(self) :
        self.type = 'activation'

    # leaky ReLU as activation fct
    def forward(self, x, alpha) :
        return max(alpha * x, x)

    # leaky ReLU prime
    def backward(self, x, alpha) :
        return 1 if x > 0 else alpha

class PRelu(Function) : # parametric ReLU
    def __init__(self) :
        self.type = 'activation'

    def forward(self, x) :
        return x

    def backward(self, x) :
        return x # TO DO

# exponetial linear unit
class ELU(Function) :
    def __init__(self) :
        self.type = 'activation'

    # elu as act fct
    def forward(self, x, alpha) :
        return x if x >= 0 else alpha * np.exp(x) -1

    # elu derivative
    def backward(self, x, alpha) :
        return 1 if x > 0 else alpha * np.exp(x)



class Tanh(Function) :
    def __init__(self) :
        self.type = 'activation'

    # tanh fct as activation
    def forward(self, x) :
        return np.tanh(x)

    # tanh primitive
    def backward(self, x) :
        return 1 - np.tanh(x) ** 2

class Sigmoid(Function) :
    def __init__(self) :
        self.type = 'activation'
    
    # sigmoid fct as activation
    def forward(self, x) :
        return 1 / (1 + np.exp(-x))

    # sigmoid primitive
    def backward(self, x) :
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 - sigmoid)

class Softmax(Function) :
    def __init__(self) :
        self.type = 'activation'

    # softmax as activation fct
    def forward(self, x) :
        expo = np.exp(x)
        expo_sum = np.sum(expo)
        return expo / expo_sum

    # softmax primitive
    def backward(self, x) :
        expo = np.exp(x)
        expo_sum = np.sum(expo)
        s = expo / expo_sum

        # init a Jacobian matrix
        jacobian_m = np.diag(s)
        for i in range(len(jacobian_m)) :
            for j in range(jacobian_m) :
                if i == j :
                    jacobian_m[i][j] = s[i] * (1 - s[i])
                else: 
                    jacobian_m[i][j] = -s[i] * s[j]
        return jacobian_m




"""-------------------------------------------------  OPTIMIZERS  -------------------------------------------------"""


class SGD(Optimizer) :
    def __init__(self, params, learning_rate = 0.001, weight_decay = 0.0, momentum = 0.9) :
        super().__init__(params)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.params = params
        self.momentum = momentum
        self.velocity = []
    # init velocity array
        for param in self.params :
            self.velocity.append(np.zeros_like(param.grad))

    # define size of steps in method of gradient decent
    def step(self) :
        for param, velocity in zip(self.params, self.velocity) :
            velocity = self.momentum * velocity + param.grad + self.weight_decay * param.data
            param.data = param.data - self.learning_rate * velocity
