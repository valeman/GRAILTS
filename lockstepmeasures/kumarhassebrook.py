import numpy as np

def kumarhassebrook(x,y):
    if len(x) != len(y):
        return -1;
    return np.sum(np.multiply(x,y)) /np.subtract(np.add(np.sum(np.square(x)),np.sum(np.square(y))),np.sum(np.multiply(x,y)))
