
# image compresion int a tensor Y then image decompresion
# separating both parts an image compreser and decompreser are built
# but during this process a lot of data is lost
# autoencoding is worth for denoising and dimension reduction

# coded in utf8

# data link in MNIST : wget http://deeplearning.net/data/mnist/mnist.pkl.gz
# compilation : python autoencoder.py 100 -e 100 -b 20 -v

""" ---> run it in goinfre, 100 epochs took all my perso space """

import numpy as np
import argparse
import cPickle as pickle
import sys
import utils

class Autoencoder(object) :
    def __init__(self, n_visible = 784, n_hidden = 784, \
        W1 = None, W2 = None, b1 = None, b2 = None,
        noise = 0.0, untied = False) :
        
        self.rng = np.random.RandomState(1)

        r = np.sqrt(6.0 / (n_hidden + n_visible + 1))

        # init randomlly our weight
        if W1 == None :
            self.W1 = self.random_init(r, (n_hidden, n_visible))

        if W2 == None :
            if untied :
                W2 = self.random_init(r, (n_visible, n_hidden))
            else :
                W2 = self.W1.T

        self.W2 = W2

        if b1 == None:
            self.b1 = np.zeros(n_hidden)
        if b2 == None:
            self.b2 = np.zeros(n_visible)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.alpha = 0.1 #learning rate
        self.noise = noise
        self.untied = untied

    # return an array with random values in range n with a size  
    def random_init(self, r, size) :
        return np.array(self.rng.uniform(low = -r, high = r, size = size))

    def sigmoid(self, x) :
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_p(self, x) :
        return x * (1.0 - x)
    
    def corrupt(self, x, noise) :
        return self.rng.binomial(size = x.shape, n = 1, p = 1.0 - noise) * x

    def encode(self, x) :
        return self.sigmoid(np.dot(self.W1, x) + self.b1)

    def decode(self, x) :
        return self.sigmoid(np.dot(self.W2, x) + self.b2)

    # have to find this math expresion and name and add to lib
    def get_cost(self, x, z) :
        eps = 1e-10
        return - np.sum((x * np.log(z + eps) + (1.0 - x) * np.log(1.0 - z + eps)))

    # return the direction to follow to descend in the gradient
    def get_cost_and_grad(self, x_batch, dnum) :
        cost = 0.0
        grad_W1 = np.zeros(self.W1.shape)
        grad_W2 = np.zeros(self.W2.shape)
        grad_b1 = np.zeros(self.b1.shape)
        grad_b2 = np.zeros(self.b2.shape)

        for x in x_batch :
            tilde_x = self.corrupt(x, self.noise)
            p = self.encode(tilde_x)
            y = self.decode(p)
            
            cost += self.get_cost(x, y)

            delta1 = - (x - y)

            if self.untied :
                grad_W2 += np.outer(delta1, p)
            else :
                grad_W1 += np.outer(delta1, p).T
            
            grad_b2 += delta1

            delta2 = np.dot(self.W2.T, delta1) * self.sigmoid_p(p)
            grad_W1 += np.outer(delta2, tilde_x)
            grad_b1 += delta2

        cost /= len(x_batch)
        grad_W1 /= len(x_batch)
        grad_W2 /= len(x_batch)
        grad_b1 /= len(x_batch)
        grad_b2 /= len(x_batch)

        return cost, grad_W1, grad_W2, grad_b1, grad_b2

    def train(self, X, epochs = 15, batch_size = 20) :
        batch_num = len(X) / batch_size

        for epoch in range(epochs) : # times to train
            total_cost = 0.0
            for i in range(batch_num) :
                batch = X[i * batch_size : (i + 1) * batch_size]

                # take gradient low dir
                cost, gradW1, gradW2, gradb1, gradb2 = \
                    self.get_cost_and_grad(batch, len(X))
                
                # just for display
                total_cost += cost

                # update weights and bias
                self.W1 -= self.alpha * gradW1
                self.W2 -= self.alpha * gradW2
                self.b1 -= self.alpha * gradb1
                self.b2 -= self.alpha * gradb2

                grad_sum = gradW1.sum() + gradW2.sum() + gradb1.sum() + gradb2.sum()

            print epoch
            print (1.0 / batch_num) * total_cost

    # dump final weight and bios
    def dump_weights(self, save_path) :
        with open(save_path, 'w') as f :
            d = {
                    "W1" : self.W1,
                    "W2" : self.W2,
                    "b1" : self.b1,
                    "b2" : self.b2,
                }

            pickle.dump(d, f)

    # display 
    def visualize_weights(self) :
        tile_size = (int(np.sqrt(self.W1[0].size)), int(np.sqrt(self.W1[0].size)))

        panel_shape = (10, 10)
        return utils.visualize_weights(self.W1, panel_shape, tile_size)

# python main 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_hidden", type = int)
    parser.add_argument("-e", "--epochs", type = int, default = 15)
    parser.add_argument("-b", "--batch_size", type = int, default = 20)
    parser.add_argument("-n", "--noise", type=float, choices=[i/10. for i in xrange(11)], default = 0.0)
    parser.add_argument('-o', '--output', type = unicode)
    parser.add_argument('-v', '--visualize', action = "store_true")
    parser.add_argument('-u', '--untied', action = "store_true")
    args = parser.parse_args()

    train_data, test_data, valid_data = utils.load_data()

    ae = Autoencoder(n_hidden = args.n_hidden, noise = args.noise, untied = args.untied)

    try:
        ae.train(train_data[0], epochs = args.epochs, batch_size = args.batch_size)
    except KeyboardInterrupt:
        exit()
        pass

    save_name = args.output
    if save_name == None:
        save_name = '%sh%d_e%d_b%d_n%d'%(
            'untied_' if args.untied else 'tied_',
            args.n_hidden, 
            args.epochs,
            args.batch_size,
            args.noise*100, 
            )

    img = ae.visualize_weights()
    img.save(save_name + ".bmp")
    if args.visualize:    
        img.show()
    ae.dump_weights(save_name + '.pkl')

