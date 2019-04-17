#!/bin/python3

import numpy as np
import math

def solve(coeffs, consts) :
    # Make sure these are the right size.
    assert(len(coeffs) == len(consts))

    curr = 0
    out_coeffs = coeffs.copy()
    out_consts = consts.copy()
    # Do Gaussian elimination.
    for i in range(min(len(out_coeffs), len(out_coeffs[0]))) :
        # Find the absolute biggest element.
        big = 0
        big_ind = 0
        for j in range(0, len(out_coeffs)) :
            if abs(out_coeffs[j][i]) > abs(big) :
                big_ind = j
                big = out_coeffs[j][i]
        if big == 0 :
            continue
        # Swap so that big_ind is first.
        for j in range(len(out_coeffs[0])) :
            out_coeffs[curr][j], out_coeffs[big_ind][j] = \
                             out_coeffs[big_ind][j], out_coeffs[curr][j]
        for j in range(len(out_consts[0])) :
            out_consts[curr][j], out_consts[big_ind][j] = \
                             out_consts[big_ind][j], out_consts[curr][j]

        # Scale
        for j in range(len(out_coeffs[0])) :
            out_coeffs[i][j] /= big
        for j in range(len(out_consts[0])) :
            out_consts[i][j] /= big

        # Eliminate
        for j in range(len(out_coeffs)) :
            if j == curr :
                continue
            scale = out_coeffs[j][i]
            for k in range(len(out_coeffs[0])) :
                out_coeffs[j][k] -= scale * out_coeffs[curr][k]
            for k in range(len(out_consts[0])) :
                out_consts[j][k] -= scale * out_consts[curr][k]
        curr += 1
    return np.array(out_coeffs), np.array(out_consts)

def gradient(ys, xs, order = 1) :
    """
    Compute the gradient at each of a number of points. This is a
    multidimensional gradient, so xs can be a list of single values or
    a list of vectors. Can compute any order up to the number of y values
    less one.
    """
    if hasattr(xs[0], "__getitem__") :
        # We're trying to solve this problem:
        # We can calculate the directional derivatives approximately,
        # so using this we can try to find a solution to
        # grad(f) * (a₁ v₁ + a₂ v₂ + …) = grad(f) * e
        # which is simply
        # a₁ (v₁ * grad(f)) + a₂ (v₂ * grad(f)) + … =
        # a₁ d₁ + a₂ d₂ + … = grad(f) * e
        # which gives us an expression to use to find each component of the
        # gradient.
        
        outs = []
        xs = np.array(xs)
        ys = np.array(ys)
        # Set the initial gradient to the zero vector.
        dirders = [(ys[i] - ys[0]) / np.linalg.norm(xs[i] -
                                                    xs[0])
                   for i in range(1, len(xs))]
        # Find the unit difference vectors.
        diffs = np.array([(xs[i] - xs[0]) / np.linalg.norm(xs[i] - xs[0]) for i in range(1, len(xs))])
        # Assuming this is a complete basis, find the scalar multiples for each
        # of the axis vectors.
        sol, _ = solve(diffs.T, np.eye(len(diffs)))
        # Using these multipliers, find the gradient for each axis. This gives
        # the complete gradient vector.
        grad = [np.dot(dirders, s) for s in sol.T]
                            
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
    
