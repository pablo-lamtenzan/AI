
# max pooler using a size of 2

import numpy as np

class MaxPool2() :
    def __init__(self) :
        None

    # generates a non overlapping 2x2 image regions to pool over
    # image is a 2d array
    def iterative_regions(self, image) :
        h, w = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h) :
            for j in range(new_w) :
                im_region = image[(i * 2) : (i * 2 + 2), (j * 2) : (j * 2 + 2)]
                yield im_region, i , j

    # return a 3d np array with dim (h / 2, w / w, nb_filters)
    # input is a 3d np array as (h, w, nb_filters)
    def forward(self, X) :
        self.last_input = X

        h, w, nb_filetrs = X.shape
        Y = np.zeros ((h // 2, w // 2, nb_filetrs))

        for im_region, i , j in self.iterative_regions(X) :
            Y[i, j] = np.amax(im_region, axis = (0, 1))

        return Y
    
    # return loss grad fot layer input 
    # dE_dY is the loss grad for this layer output
    def backward(self, dE_dY) :

        dE_dX = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterative_regions(self.last_input) :
            h, w, f = im_region
        amax = np.amax(im_region, axis = (0, 1))

        for i2 in range(h) :
            for j2 in range(w) :
                for f2 in range(f) :
                    #if pixel has max val cp grad to it 
                    if im_region[i2, j2, f2] == amax[f2] :
                        dE_dX[i * 2 + i2, j * 2 + j2, f2] = dE_dY[i, j, f2]

        return dE_dX