import math
import numpy as np;

def dist(x,y):
    return (x - y) ** 2;

def twed_scb(x,timesx,y,timesy,lamb,nu,slope=5):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));


    for i in range(len(x)):
        temp = prev;
        prev = cur;
        cur = temp;
        minw =  np.ceil(max(min_slope * i,
                    ((ylen-1) - max_slope * (xlen - 1)
                        + max_slope * i)))
        maxw = np.floor(min(max_slope * i,
                   ((ylen - 1) - min_slope * (xlen - 1)
                      + min_slope * i)) + 1);
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

