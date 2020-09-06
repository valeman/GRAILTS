import numpy as np

def jeffrey(x,y):
    if len(x) != len(y):
        return -1;
    return np.sum(np.multiply((np.subtract(x,y)),np.log(np.divide(x,y))));

    return sum;
