# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    estimate_price.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: plamtenz <plamtenz@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/31 07:11:03 by plamtenz          #+#    #+#              #
#    Updated: 2020/01/31 08:23:27 by plamtenz         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import math as mt
import csv
import sys
import os

from training import Train

# inherit from Train Class propreties and methods
class Estimate_price(Train) :
    def __init__(self) :
        self.mileages = None
        self.t0 = 0.0
        self.t1 = 0.0
        self.theta_path = theta_path
    
    # open thetas file and read it, then store values in to self.t0 self.t1
    def read_thetas(self, theta_path) :
        if os.path.isfile(theta_path) :
            with open(theta_path, 'r') as thetasfile :
                ret = csv.reader(thetasfile, delimiter = ',')
                """ if path to file exist, open and name it, then read it"""
                for line in file :
                    self.t0 = float(line[0])
                    self.t1 = float(line[1])
                    break # this break and this for seems not necesarry
    
    # user interact with the program givin a mileage
    # this fct check i fis valid and returns it
    def get_data_from_user(self) :
        while 42 :
            print('Milege : ')
            try :
                mileage = input()
            except EOFError :
                sys.exit('EOF input. Progam exit with status. ')
            except :
                sys.exit('Unknown Error. Progam exit with status. ')
            try : 
                mileage = int(mileage)
                if mileage < 0 :
                    print('Mileage value must be a positive integer. ')
                else :
                    break
            except ValueError :
                print('Not a value value for the mileage. ')

    # estimated price of mileage = t0 + (t1 * mileage)
    def price_estimation(self, prices) :
        final_price = t0 + (self.t1 * self.normalize(self.mileages, self.mileage)
        return (self.denormalize(prices, final_price))
        


if __name__ == "__main__" :
    
    x = Estimate_price()
    
    # 1st thetas values are searched from thetas file 
    x.read_thetas('thetas.csv')
    # 2nd ask to user to put some mileages values
    x.get_data_from_user()
    # 3rd get all data to have some values to learn and correct the error
    mileages, prices = getDATA('data.csv')
    # 4th calc price 
    price = mt.ceil(x.price_estimation(prices)) # ceil(x) return smallest integral >= to x
    
    # 5th estimate woth
    if price <= 0 :
        price = 0
        print ('This shit doesn\'t worth nothing!\n Come back with better material. ')
    else :
        print('This mileage worths : %d\n' % price)