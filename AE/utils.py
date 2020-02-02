
# utils for AE

import numpy as np
import cPickle as pickle
import gzip
# import Image
from PIL import Image

def load_data() :
    with gzip.open('mnist.pkl.gz') as f :
        tr, te, vl, = pickle.load(f)
    return tr, te, vl

def visualize_weights(weights, panel_shape, tile_size) :

    def scale(x):
        eps = 1e-8
        x = x.copy()
        x -= x.min()
        x *= 1.0 / (x.max() + eps)
        return 255.0 * x
    
    margin_y = np.zeros(tile_size[1])
    margin_x = np.zeros((tile_size[0] + 1) * panel_shape[0])
    image = margin_x.copy()

    for y in range(panel_shape[1]):
        tmp = np.hstack( [ np.c_[ scale( x.reshape(tile_size) ), margin_y ] 
            for x in weights[y*panel_shape[0]:(y+1)*panel_shape[0]]])
        tmp = np.vstack([tmp, margin_x])

        image = np.vstack([image, tmp])

    img = Image.fromarray(image)
    img = img.convert('RGB')
    return img

