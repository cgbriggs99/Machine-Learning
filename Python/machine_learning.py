#!/bin/python3

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
          / len(ins) ** 2,
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
    assert(len(inputs) == len(outputs))
    if type(weight_guess) == list :
        w1 = np.array(weight_guess)
    else :
        w1 = weight_guess.copy()
    w2 = None
    wdiff = np.array([2 * random.random() - 1 for i in w1])
    try :
        g1 = gradient(lambda w : loss(regression, w, inputs, outputs), w1, wdiff)
    except Exception as err:
        print("Error! Bad division!")
        print(f"w1: {w1}\nwdiff: {wdiff}")
        print(err.__traceback__)
        raise(err)
    g2 = None
    w2 = w1
    w1 = w1 - 0.001 * g1;
    g2 = g1;
    try :
        g1 = gradient(lambda w : loss(regression, w, inputs, outputs), w1, w1 - w2)
    except Exception as err:
        print("Error! Bad division!")
        print(f"w1: {w1}\nw2: {w2}\ng1: {g1}\ng2: {g2}")
        print(err.__traceback__)
        raise(err)
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
        try :
            g1 = gradient(lambda w : loss(regression, w, inputs, outputs),
                          w1, w1 - w2)
        except Exception as err :
            print("Error! Bad division!")
            print(f"w1: {w1}\nw2: {w2}\ng1: {g1}\ng2: {g2}\ncount: {count}")
            print(err.__traceback__)
            raise(err)
        count += 1
    return w1

def cross_validation(regression, weight_guess, inputs, outputs, k = 10,
                     gradient = lambda f, w1, wdiff :
          np.array([(f([w1[j] + (wdiff[j] if i == j else 0) for j in
                        range(0, len(w1))])
            - f(w1)) / wdiff[i] for i in range(0, len(w1))]),
          loss = lambda f, w, ins, outs :
          sum([(f(w, ins[i]) - outs[i]) ** 2 for i in range(0, len(ins))])
          / len(ins) ** 2,
          step = 0.001, eps = 0.00001) :
    ins = inputs
    outs = outputs
    rem_ins = np.array_split(ins, k)
    rem_outs = np.array_split(outs, k)
    weights = []
    weight = weight_guess
    mles = []
    mle = 0
    for i in range(k) :
        train_in = [ins[j] for j in range(len(ins)) if list(rem_ins[i]).count(ins[j]) == 0]
        train_out = [outs[j] for j in range(len(outs)) if list(rem_outs[i]).count(outs[j]) == 0]
        val_ins = rem_ins[i]
        val_outs = rem_outs[i]
        weight = teach(regression, weight, train_in, train_out, gradient, loss, step, eps)
        weights.append(weight)
        mle = loss(regression, weight, val_ins, val_outs)
        mles.append(mle)
    return weight, mle, weights, mles
        

#Run tests
def main() :
    import matplotlib.pyplot as plt
    import matplotlib.markers as mark
    from mpl_toolkits.mplot3d import Axes3D as plt3d

    #np.seterr(all="raise")

    r = lambda w, in_ : w[0] * in_[0] + w[1]
    r2 = lambda w, in_ : w[0] * in_[0] ** 2 + w[1] * in_[1] ** 2 + \
         w[2] * in_[0] * in_[1] + w[3] * in_[0] + w[4] * in_[1] + w[5]
    
    input1 = [[0], [1], [2], [3], [4]]
    input4 = [[random.random(), random.random()] for i in range(500)]
    output1 = [0, 1, 2, 3, 4]
    output2 = [1, 3, 5, 7, 9]
    output3 = [0]
    weight4 = [random.random() for i in range(6)]
    output4 = list(map(lambda x : r2(weight4, x), input4))


    bot1 = Bot(r, [1, 0])
    bot2 = Bot(r, [2, 1])
    bot4 = Bot(r2, weight4)

    bot_test_1 = Bot(r, cross_validation(r, [0, 0], input1, output1, k = 3, step = 0.00001, eps = 0.000001)[0])
    bot_test_2 = Bot(r, teach(r, [0, 0], input1, output2))
    bot_test_4 = Bot(r2, teach(r2, [0 for i in range(6)],
                               input4, output4, eps = 0.0001))
    try :
        bot_test_3 = Bot(r, teach(r, [0, 0], input1, output3))
    except AssertionError :
        pass;
    else :
        print("Failed")
        s = input("Press any key to continue")
        if s != None :
            exit()

    try :
        assert(bot_test_1.get_weights[i] == bot1.get_weights[i] for i in range(0, 2))
        assert(bot_test_2.get_weights[i] == bot2.get_weights[i] for i in range(0, 2))
    except AssertionError :
        print("Failed.")
    else :
        print("Succeeded")
    plt.figure()
    plt.title("Bot 1")

    avg = sum(output2) / len(output2)
    sum_exp = sum((o - avg) ** 2 for o in output1)
    sum_reg = sum((bot_test_2.compute(i) - avg) ** 2 for i in input1)
    
    plt.scatter([input1[i][0] for i in range(len(input1))], output1,
                marker = mark.MarkerStyle("o"))
    plt.plot([input1[i][0] for i in range(len(input1))],
             [bot1.compute(i) for i in input1], "b--", label = "Expected Bot")
    plt.plot([input1[i][0] for i in range(len(input1))],
             [bot_test_1.compute(i) for i in input1], "r--",
             label = f"Computed Bot, w = {bot_test_1.get_weights()}, r² = {sum_reg / sum_exp}")
    plt.legend()

    
    plt.figure()

    avg = sum(output2) / len(output2)
    sum_exp = sum((o - avg) ** 2 for o in output2)
    sum_reg = sum((bot_test_2.compute(i) - avg) ** 2 for i in input1)
    
    plt.title("Bot 2")
    plt.scatter([input1[i][0] for i in range(len(input1))], output2,
                marker = mark.MarkerStyle("o"))
    plt.plot([input1[i][0] for i in range(len(input1))],
             [bot2.compute(i) for i in input1], "b--", label = "Expected Bot")
    plt.plot([input1[i][0] for i in range(len(input1))],
             [bot_test_2.compute(i) for i in input1], "r--",
             label = f"Computed Bot, w = {bot_test_2.get_weights()}, r² = {sum_reg / sum_exp}")
    plt.legend()
    plt.figure()

    avg = sum(output4) / len(output4)
    sum_exp = sum((o - avg) ** 2 for o in output4)
    sum_reg = sum((r2(bot_test_4.get_weights(), input4[i]) - avg) ** 2
                  for i in range(len(output4)))
    ax = plt.gca(projection="3d")
    ax.scatter(np.array([v[0] for v in input4]),
                  np.array([v[1] for v in input4]), np.array(output4))
    X, Y = np.meshgrid([v[0] for v in input4], [v[1] for v in input4])
    Z = np.array([[r2(bot_test_4.get_weights(), [X[i][j], Y[i][j]])
          for j in range(len(X[i]))] for i in range(len(X))])
    ax.plot_surface(X, Y, Z, rcount=25, ccount=25)
    plt.title(f"Bot Calculated: r² = {sum_reg / sum_exp}")
    plt.show()

if __name__ == "__main__" :
    main()
