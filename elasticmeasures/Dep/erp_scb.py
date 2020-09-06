import numpy as np
import math

def erp_scb(x,y,m,w):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));

    for i in range(len(x)):
        minw = max(0,i-w);
        maxw = min(len(y), i + w);
        temp = prev;
        prev = cur;
        cur = prev;

        for j in range(int(minw),int(maxw)):
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = cur[j-1] - pow(y[j] - m,2);
            elif j == wmin:
                cur[j] = prev[j] - pow(x[i] - m, 2);
            else:
                cur[j] = max(prev[j-1] - pow(x[i] - y[j],2), cur[j-1] - pow(y[j] - m,2), prev[j] - pow(x[i] - m,2));
        
    return math.sqrt(0 - cur[len(y)-1]);



def erpabs_scb(x,y,m,w):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));

    for i in range(len(x)):
        minw = max(0,i-w);
        maxw = min(len(y), i + w);

        for j in range(int(minw),int(maxw)):
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = cur[j-1] - abs(y[j] - m);
            elif j == wmin:
                cur[j] = prev[j] - abs(x[i] - m);
            else:
                cur[j] = max(prev[j-1] - abs(x[i] - y[j]), cur[j-1] - abs(y[j] - m), prev[j] - abs(x[i] - m));

        temp = prev;
        prev = cur;
        cur = temp;

    return math.sqrt(0 - cur[len(y)-1]);

