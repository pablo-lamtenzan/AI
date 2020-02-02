
import numpy as np

'''

note from tutorial:

    Note: In this implementation, we assume the input is a 2d numpy array for simplicity, because that's
    how our MNIST images are stored. This works for us because we use it as the first layer in our
    network, but most CNNs have many more Conv layers. If we were building a bigger network that needed
    to use Conv3x3 multiple times, we'd have to make the input be a 3d numpy array.
'''

# conv using 3x3 filters

class Conv3x3() :
    def __init__(self, num_filters) :
        self.num_filters = num_filters
        # divide filters by to reduce variance
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    # geretates all possible 3x3 image regions using valid padding
    # image is a 2d np array
    def iterative_regions(self, image) :
        
        w, h = image.shape

        for i in range(h - 2) :
            for j in range(w - 2) :
                im_regions = image[i : (i + 3), j : (j + 3)]
                yield im_regions, i, j # returns  a genetator

    # return a 3d np array with dims (h, w, nb filters)
    # input is a 2d np array
    def forward(self, input) :
        self.last_input = input # save input
        h, w = input.shape
        Y = np.zeros((h - 2, w - 2, self.num_filters))

        for im_regions, i, j in self.iterative_regions(input) :
            Y[i, j] = np.sum(im_regions * self.filters, axis = (1, 2))

        return Y

    # dE_dY is the loss gradient for layer output
    def bakward(self, dE_dY, learning_rate = 0.05) :
        dE_dFilters = np.zeros(self.num_filters)

        for im_region, i, j in self.iterative_regions(self.last_input) :
            for f in range(self.num_filters) :
                dE_dFilters[f] += dE_dY[i, j, f] * im_region

        # update filters
        self.filters -= learning_rate * dE_dFilters

        # in this specific nothing is returned cause its ussed as firts layer at CNN
        # Otherwise must ret loss grad like every layer in CNN
        return None
