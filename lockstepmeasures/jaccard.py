import numpy as np

def jaccard(x,y):
    if len(x) != len(y):
        return -1;
    return np.sum(np.square(np.subtract(x,y))) / np.sum(np.square(x) + np.square(y) + np.multiply(np.add(x,y),-1));
