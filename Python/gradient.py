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
    if hasattr(xs[0], "__getitem__") :
        outs = []
        for i in range(0, len(ys)) :
            grad = []
            outs.append(np.array(grad))
        return outs
    else :
        outs = []
        for j in range(0, len(ys)) : 
            const_array = np.array([0 for i in range(0, len(ys))])
            const_array[1] = math.factorial(i)
            coeff_array = \
                    np.array([[(xs[l] - xs[j]) ** m for l in range(0, len(xs))]
                              for m in range(0, len(xs))])
            sol = np.linalg.solve(coeff_array, const_array)
            outs.append(sum(sol[i] * ys[i] for i in range(0, len(sol))))
        return (outs)
    
