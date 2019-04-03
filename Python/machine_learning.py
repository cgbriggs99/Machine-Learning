import math
import numpy as np
import random

class Bot :
    """
Represents a bot to be taught.
"""
    #Initialize the class.
    def __init__(self, regression, num_weights) :
        self.__regression = regression
        if hasattr(num_weights, "__len__") :
            self.__weights = np.array(num_weights)
        else :
            self.__weights = np.zeros(num_weights)

    #Getters and setters
    def get_regression(self) :
        return (self.__regression)
    def get_weights(self) :
        return (self.__weights.copy())

    def set_regression(self, regression) :
        self.__regression = regression
    def set_weights(self, weights) :
        self.__weights = weights.copy()

    def compute(self, input_) :
        return (self.__regression(self.__weights, input_))
    
def teach(regression, weight_guess, inputs, outputs,
          gradient = lambda f, w1, wdiff :
          np.array([(f([w1[j] + (wdiff[j] if i == j else 0) for j in
                        range(0, len(w1))])
            - f(w1)) / wdiff[i] for i in range(0, len(w1))]),
          loss = lambda f, w, ins, outs :
          sum([(f(w, ins[i]) - outs[i]) ** 2 for i in range(0, len(ins))])
          / len(ins[i]) ** 2,
          step = 0.001, eps = 0.00001) :
    """
Teaches a bot with a given regression and initial weight against a set of
inputs and outputs.\n
regression(weights, input): A function that takes in weights and an input. \n
weight_guess: The initial guess array for the weights. Typically all zero, but
the function needs the size.\n
inputs: A list of lists of values to use as test inputs.\n
outputs: A list of values to be used as outputs.\n
gradient: A function that takes a loss function, a weight vector, and a vector
containing the changes in each weight component and returns the gradient.
Defaults to a first-order difference method, but may be replaced as most useful\n
loss: The loss function. This calculates the total differences between the
bot's outputs and the expected outputs. Defaults to the difference-squared loss
function.\n
step: The steps to take at each iteration. Can be a value or can be callable.
In the case that it is callable, use the signature of
step(wn, w(n-1), grad(n), grad(n-1)).\n
eps: The convergence. Will also loop until 1/eps steps have been taken, so as to
not be looping forever.
"""
    if type(weight_guess) == list :
        w1 = np.array(weight_guess)
    else :
        w1 = weight_guess.copy()
    w2 = None
    wdiff = np.array([random.random() for i in w1])
    g1 = gradient(lambda w : loss(regression, w, inputs, outputs), w1, wdiff)
    g2 = None
    w2 = w1
    w1 = w1 - 0.001 * g1;
    g2 = g1;
    g1 = gradient(lambda w : loss(regression, w, inputs, outputs), w1, w1 - w2)
    count = 0
    while (math.sqrt((w1 - w2).dot(w1 - w2)) / (len(w1) ** 2) > eps or
           g1.dot(g1) > eps or g2.dot(g2) > eps) and count < 1/eps :
        if hasattr(step, "__call__") :
            change = step(w1, w2, g1, g2)
        else :
            change = step
        w2 = w1
        g2 = g1
        w1 = w1 - change * g1
        g1 = gradient(lambda w : loss(regression, w, inputs, outputs),
                      w1, w1 - w2)
        count += 1
    return Bot(regression, w1)

