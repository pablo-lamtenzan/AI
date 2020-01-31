# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    test.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: plamtenz <plamtenz@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/31 08:34:18 by plamtenz          #+#    #+#              #
#    Updated: 2020/01/31 10:58:21 by plamtenz         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import maht as mt
import csv
import matplotlib.pyplot as plt
import tools 

class Train :
    def __init__(self) :
        self.t0 = 0.0
        self.t1 = 0.0
        self.mileages = []
        self.prices = []
    
    # open a given file an split ',' and store data
    def get_data_from_file(self, file_path) :
        with open(file_path, 'r') as csvfile :
            file = csv.reader(csvfile, delimiter = ',')
            for split in file :
                self.mileages.append(split[0])
                self.prices.append(split[1])
        
        # remove titles of the data
        self.mileages.pop(0)
        self.prices.pop(0)

        # eval lest python run Python code within itself
        for i in range(0, len(self.mileages)) :
            self.mileages[i] = eval(self.mileages[i])
            self.prices[i] = eval(self.prices[i])
        """ nor sure if i need last loop"""

    # used proportionality to give a x nb of n range into a y nb in range [1 ; -1]
    # into a given array
    def normalize(self, x, array) :
        return ((x - min(array) / max(array) - min(array)))

    # inverse operation 
    def denormalize(self, x, array) :
        return ((x * (max(array) - min(array)) + min(array)))

    # return 2 normalizated array of the data
    def normalizeData(self) :
        x = [] # normalized mileages array
        y = [] # normalized prices array
        
        # normalize x
        for mileage in self.mileages :
            x.append(self.normalize(mileage, self.mileages))

        # normalize y
        for price in self.prices :
            y.append(self.normalize(price, self.prices))

        return x, y

    # (1 / n) * (sum(y_pred - y_true) ** 2)
    # used to calc error rate
    def cuadratic_mean_error(self) :
        loss_rate = 0.0
        # zip(x, y) join x and y as a same elem during an iteration
        for mileage, price in zip(self.mileages, self.prices) :
             loss_rate += (price - (self.t0 + (self.t1 * mileage))) ** 2
             return (loss_rate / len(self.mileages))

    # used to optimise error rate
    def gradientDescent(self, learning_rate, epochs) :
        lossHistory = []
        t0History = [0.0]
        t1History = [0.0]

        for i in range(epochs) :
            dt0 = 0.0
            dt1 = 0.0
            for mileage, price in zip(self.mileages, self.prices) :
                # calc temp using subject fcts
                dt0 += (self.t1 * mileage + self.t0) - price
                dt1 += ((self.t1 * mileage + self.t0) - price) * mileage

            # uptate our thetas
            self.t0 -= dt0 / len(mileages) * learning_rate
            self.t1 -= dt1 / len(prices) * learning_rate

            # calc error rate
            loss = self.cuadratic_mean_error()
            learning_rate = boldDriver(loss, lossHistory, dt0, dt1, learning_rate, len(mileages))
            lossHistory.append(loss)
            t0History.append(t0)
            t1History.append(t1)

        return lossHistory, t0History, t1History

    def boldDriver(self, loss, lossHistory, dt0, dt1, learning_rate, size) :
        new_learning_rate = learning_rate
        if len(lossHistory) > 1 :
            # if loss_rate > lass loss history last added, update thetas and decrease lr
            if loss >= lossHistory[-1] :
                self.t0 += dt0 / size * learning_rate
                self.t0 += dt1 / size * learning_rate
                new_learning_rate *= 0.5
            # else are far the decent zone, must increase step size
            else :
                new_learning_rate *= 1.05
        
        return new_learning_rate

    def store_file_data(self, file_path) :
        # open with write rights
        with open(file_path, 'w') as csvfile :
            wfile = csv.writer(csvfile, demiliter = ',', quotechar = '"', quoting = cvs.QUOTE_MINIMAL)
            wfile.writerow([self.t0, self.t1])

    def display_graph(self, lossHistory, t0History, t1History) :
        x = [float(min(self.mileages)), float(max(mileages))]
        y = []

        # take mean
        for elem in x :
            elem = t1 * self.normalize(self.mileages, elem) + self.t0
            y.append(self.denormalize(elem))

        plt.figure(1) # new figure management
        plt.plot(self.mileages, self.prices, 'bo', x, y, 'r-') # tracks data into current figure
        plt.xlabel('mileage') # name x axis
        plt.ylabel('price') # name y axis

        plt.figure(2)
        plt.plot(lossHistory, 'r.') # 'r.' is color
        plt.xlabel('iterations')
        plt.ylabel('loss')

        plt.figure(3)
        plt.plot(t0History, 'g.')
        plt.xlabel('iterations')
        plt.ylabel('t0')

        plt.figure(4)
        plt.plot(t0History, 'b.')
        plt.xlabel('iterations')
        plt.ylabel('t1')

        plt.show()

if __name__ == '__main__' :

    x = Train
    learning_rate = 0.1 # can put this values in __init__()
    epochs = 100

    # 1st parse data of the given file
    x.get_data_from_file('data.csv')
    # 2nd normalize this data
    self.mileages, self.price = x.normalizeData() """ can rewrite this better in fct """
    # calc error and optimize
    lossHistory, t0History, t1History = x.gradientDescent(learning_rate, epochs)
    # store opimized thetas
    x.store_file_data('thetas.csv')
    # display graphs with learning stats
    x.display_graph(lossHistory, t0History, t1History)