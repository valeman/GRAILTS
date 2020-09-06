import numpy as np;

def zrep(x,y):
    return (np.sum(np.subtract(x[i],y[i]))** (1/2));