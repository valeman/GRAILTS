import numpy as np


def MinMaxNorm(x,alpha,beta):

    minx = np.min(x);
    maxx = np.max(x);

    return np.add(np.divide(np.multiply(np.subtract(x,minx),beta-alpha),maxx-minx),alpha);