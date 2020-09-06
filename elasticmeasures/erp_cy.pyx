import numpy as np
import math;
cimport numpy as np

cdef double erp(np.ndarray x,np.ndarray y, double m):
    np.insert(x,0,0);
    np.insert(y,0,0);
    df = np.zeros((len(x), len(y)))

    cdef int i=  0;
    cdef int j = 0;
    for i in range(1,len(y)):
        df[0][i] = df[0][i-1] - pow(y[i] - m, 2);

    for i in range(1,len(x)):
        df[i][0] = df[i-1][0] - pow(x[i] - m, 2);

    df[1][1] = 0;

    for i in range(1,len(x)):
        for j in range(1,len(y)):
            df[i][j] = max(df[i - 1][j-1] - pow(x[i] - y[j],2), df[i][j-1] - pow(y[j] - m,2), df[i-1][j] - pow(x[i] - m, 2));

    return math.sqrt(0 - df[len(x) - 1][len(y) -1 ]);



cdef erpabs(x,y,m):

    np.insert(x,0,0);
    np.insert(y,0,0);
    df = np.zeros((len(x), len(y)))

    cdef int i=  0;
    cdef int j = 0;
    for i in range(1,len(y)):
        df[0][i] = df[0][i-1] - abs(y[i] - m);

    for i in range(1,len(x)):
        df[i][0] = df[i-1][0] - abs(x[i] - m);

    df[1][1] = 0;

    for i in range(1,len(x)):
        for j in range(1,len(y)):
            df[i][j] = max(df[i - 1][j-1] - abs(x[i] - y[j]), df[i][j-1] - abs(y[j] - m), df[i-1][j] - abs(x[i] - m));

    return math.sqrt(0 - df[len(x) - 1][len(y) -1 ]);