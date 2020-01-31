
# Network Class 

class Network :
    def __init__(self) :
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer) :
        self.layers.append(layer)
    
    # set loss to use
    def use(self, loss, loss_prime) :
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for a given input
    def predict(self, input_data) :
        # sample dimension first
        samples = len(input_data)
        result = []

        #run network over all samples using forward propagation
        for i in range(samples) :
            output = input_data[i]
            for layer in self.layers :
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    # train Network
    def fit(self, x_train, y_train, epochs, learning_rate) :
        # sample dimension first
        samples = len(x_train)

        # trainig loop
        for i in range(epochs) :
            err = 0
            for j in range(samples) :

                # forward propagation
                output = x_train[j]
                for layer in self.layers :
                    output = layer.forward_propagation(output)

                # compute loss only for display reasons
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers) :
                    error = layer.backward_propagation(error, learning_rate)

                # display
                err /= samples
                print('epoch %d %d      error=%f' % (i + 1, epochs, err))

