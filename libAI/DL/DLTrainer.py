
import numpy as np
import matplotlib.pyplot as plt
import DLUtils

# used to encapsulate functionalities in a class
# a computation graph

""" trainig cicle :

model ->        computation graph
data,target ->  training data
loss_fn ->      optimization objective
optim ->        optimizer to update model parameters to minimize loss


Repeat :        (until convergence or for predefined number of epochs)

    optim.zeroGrad() ->                 set all gradients  to zero
    output = model.forward(data) ->     get output from  model
    loss = loss_fn(output,target) ->    calculate loss
    grad = loss.backward() ->           calculate gradient of loss w.r.t output
    model.backward(grad) ->             calculate gradients for all the parameters
    optim.step() ->                     update model parameters

"""

class Model() :
    def __init__(self) :
        self.computation_graph = []
        self.params = []

    # add layer to model
    def add(self, layer) :
        self.computation_graph.append(layer)
        self.params += layer.getParams()

    # init weight and bias
    def initNetwork(self) :
        for f in self.computation_graph :
            if f.type == 'linear' :
                weights, bias = f.getParams()
                weights.data = 0.01 * np.random.randn(weights.data.shape[0], weights.data.shape[1])
                bias.data = 0.0
    
    # main fct to train our ai
    def fit(self, data, target, batch_size, epochs, optimizer, loss) :
        loss_history = []
        self.initNetwork()
        data_gen = DLUtils.DataGenerator(data, target, batch_size)
        it = 0
        for epoch in range(epochs) :
            for X, Y in data_gen :
                # init 
                optimizer.zeroGrad()
                for f in self.computation_graph :
                    X = f.forward(X)
                # calc loss
                loss_rate = loss.forward(X, Y)
                # optimize rate
                grad = loss.backward()
                # display
                for f in self.computation_graph[::-1] :
                    grad = f.backward(grad)
                loss_history += [loss_rate]
                print("Loss at epoch = {} and iteration = {}: {}".format(epoch, it, loss_history[-1]))
                it += 1
                optimizer.step()
        return loss_history

    def predict(self, data) :
        X = data
        for f in self.computation_graph :
            X = f.forward(X)
        return (X)

    