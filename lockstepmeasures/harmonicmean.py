import numpy as np;

def harmonicmean(x,y):
    if len(x) != len(y):
        return -1;

    a = np.multiply(x,y);
    b = np.linalg.pinv([np.add(x,y)]);

    return 2 * np.sum(np.dot(a,b))
