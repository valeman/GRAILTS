import numpy as np
import math

def cDTW(t, r, W):
    '''

    :param t: time series
    :param r: time series
    :param W: constraint window
    :return: DTW distance
    '''
    n = len(t)
    m = len(r)
    D = np.ones((n+1, m+1)) * math.inf
    D[0,0] = 0

    for i in range(1,n+1):
        for j in range(max(1, i -W),min(m+1, i + W)):
            cost = (t[i - 1] - r[j - 1]) ** 2
            D[i, j] = cost + min([D[i - 1, j], D[i - 1, j - 1], D[i, j - 1]])

    return np.sqrt(D[n,m])
