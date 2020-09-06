import numpy as np;


def MeanNorm(x):

    minx = np.amin(x);
    maxx = np.amax(x);

    return np.divide(x,maxx-minx);