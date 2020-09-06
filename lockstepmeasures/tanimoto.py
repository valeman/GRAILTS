import numpy as np

def tanimoto(x,y):
    if len(x) != len(y):
        return -1;
    minxy = np.minimum(x,y);
    sumxy = np.sum(x) + np.sum(y);
    a = (sumxy - 2 * minxy)
    b = np.linalg.pinv([sumxy - minxy])
    return np.sum(np.dot(a,b));
