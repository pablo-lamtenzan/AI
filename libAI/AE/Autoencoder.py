
# image compresion int a tensor Y then image decompresion
# separating both parts an image compreser and decompreser are built
# but during this process a lot of data is lost
# autoencoding is worth for denoising and dimension reduction

# coded in utf8

# data link in MNIST : wget http://deeplearning.net/data/mnist/mnist.pkl.gz
# compilation : python autoencoder.py 100 -e 1 -b 20 -v

import numpy as np
import argparse
import cPickle as pickle
import utils

class Autoencoder(object) :
    def __init__(self, n_visible = 784, n_hidden = 784, \
        W1 = None, W2 = None, b1 = None, b2 = None,
        noise = 0.0, untied = False) :
    
    # return an array with random values in range n with a size  
    def random_init(self, r, size) :
        return np.array(self, )
