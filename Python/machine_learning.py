import math
import numpy as np

class Bot :
    def __init__(self, regression, num_weights) :
        self.__regression = regression
        self.__weights = np.zeros(num_weights)

    def get_regression(self) :
        return (self.__regression)
    def get_weights(self) :
        return (self.__weights.copy())

    def set_regression(self, regression) :
        self.__regression = regression
    def set_weights(self, weights) :
        self.__weights = weights.copy()
    
