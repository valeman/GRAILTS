import numpy as np

def k_divergence(x,y):
    if len(x) != len(y):
        return -1;
    for i in range(len(x)):
        if y[i] <= 0:
            return -1;
    return np.sum(np.multiply(x,np.log(np.divide(np.multiply(x,2),np.add(x,y)))));
