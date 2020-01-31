
# Layer obj

class Layer :
    def __init__(self) :
        self.input = None
        self.output = None

    # compute output Y from X layer input
    def forward_propagation(self, input) :
        raise NotImplementedError

    # compute dE / dX for a given dE / dY 
    def backward_propagation(self, output_error, learning_rate) :
        raise NotImplementedError

