import numpy as np
cimport numpy as np

cdef double msm_dist(double new, np.ndarray x, np.ndarray y, double c):
    cdef double dist = 0;
    if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
        dist = c;
    else:
        dist = c + min(abs(new - x), abs(new - y))

    return dist;

def msm(np.ndarray x,np.ndarray y, double c):

    cost = np.zeros((len(x),len(y)));

    cdef int xlen = len(x);
    cdef int ylen = len(y);

    cost[0][0] = abs(x[i] - y[i]);

    cdef int i = 0;
    cdef int j = 0;
    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c);

    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c);

    for i in range(1,xlen):
        for j in range(1,ylen):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c));

    return cost[xlen-1][ylen-1];
