#!/bin/python3

import numpy as np
import math

def gradient(ys, xs, order = 1) :
    """
    Compute the gradient at each of a number of points. This is a
    multidimensional gradient, so xs can be a list of single values or
    a list of vectors. Can compute any order up to the number of y values
    less one.
    """
    outs = []
    for i in range(0, order) :
        for j in range(0, len(ys)) :
            grad = []
            for k in range(0, len(xs[0])) :
                # Set up constant array.
                const_array = np.array([0 for i in range(0, len(ys))])
                const_array[i + 1] = math.factorial(i)

                #Set up coefficient array.
                coeff_array =
                np.array([[
