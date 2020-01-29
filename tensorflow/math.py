
# some nice maths functions

import numpy as np

"""here implement random_normal() but is hard as fck we have to check better tensorflow architecture and no onl build
    math fuction, we have to be abble to check the type for example"""

def add(x1, x2) : #normally this can work with 2 tensors
    return (np.add(x1, x2))

def sub(x1, x2) :
    return (np.subtract(x1 ,x2))

"""
def mult(x1, x2) :
    return (np.multiply(x1, x2))

def div(x1, x2) :
    return (np.divide(x1, x2))

def mod(x1, x2) :
    return (np.mod(x1, x2))

def pow(x1, p) :
    return (np.power(x1, p))
    """

def matmul(x1, x2) : # product of 2 tensors
    return (np.matmul(x1, x2))