import numpy as np;




def AdaptiveScaling(x,y):
    alpha = np.sum(np.multiply(x,y))/np.sum(np.multiply(y,y));
    ynew = np.multiply(y,alpha);

    return x, ynew;