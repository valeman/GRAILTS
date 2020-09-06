import math
import numpy as np;

def dist(x,y):
    return (x - y) ** 2;

def twed_scb(x,timesx,y,timesy,lamb,nu,w=5):

    xlen = len(x);
    ylen = len(y);
    cur = np.zeros(ylen);
    prev = np.zeros(ylen);

    for i in range(0,xlen):
        temp = prev;
        prev = cur;
        cur = temp;
        minw = max(0,i-w);
        maxw = min(ylen,i+w);
        for j in range(int(minw),int(maxw)):
            if i + j == 0:
                cur[j] = math.sqrt(dist(x[i], y[j]))
            elif i == 0:
                c1 = cur[j - 1] + math.sqrt(dist(y[j - 1], y[j])) + nu * (timesy[j] - timesy[j - 1]) + lamb;
                cur[j] = c1
            elif j == 0:
                c1 = prev[j] + math.sqrt(dist(x[i - 1], x[i])) + nu * (timesx[i] - timesx[i - 1]) + lamb;
                cur[j] = c1
            else:
                c1 = prev[j] + math.sqrt(dist(x[i - 1], x[i])) + nu * (timesx[i] - timesx[i - 1]) + lamb;
                c2 = cur[j - 1] + math.sqrt(dist(y[j - 1], y[j])) + nu * (timesy[j] - timesy[j - 1]) + lamb;
                c3 = prev[j - 1] + math.sqrt(dist(x[i], y[j])) + math.sqrt(dist(x[i - 1], y[j - 1])) + nu * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]));
                cur[j] = min(c1,c2,c3);
        
        

    return cur[ylen-1];

