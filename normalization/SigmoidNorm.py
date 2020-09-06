import numpy as np;


def SigmoidNorm(x):

    return np.reciprocal(np.subtract(1,np.exp(np.multiply(-1,x))));