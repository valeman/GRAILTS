import math
import numpy as np;
import warnings
def dist(x,y):
    return (x - y) ** 2;


def twed(x,timesx,y,timesy,lamb,numb,constraint=None,w=5):
    
    if (constraint == "None"):
        return twed_n(x,timesx,y,timesy,lamb,nu);
    elif constraint == "Sakoe-Chiba":
        return twed_scb(x,timesx,y,timesy,lamb,nu,w);
    elif constraint == "Itakura":
        return twed_ip(x,timesx,y,timesy,lamb,nu,w);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return twed_n(x,timesx,y,timesy,lamb,nu);


def twed_n(x,timesx,y,timesy,lamb,nu):

    x.insert(0,0);
    y.insert(0,0);
    timesx.insert(0,0);
    timesy.insert(0,0);
    dp = np.zeros((len(x),len(y)));
    xlen = len(x);
    ylen = len(y);

    for i in range(xlen):
      dp[i][0] = float('inf')
    for i in range(ylen):
      dp[0][i] = float('inf');
    dp[0][0] = 0;

    for i in range(1,xlen):
        for j in range(1,ylen):
             c1 = dp[i - 1][j] + math.sqrt(dist(x[i - 1], x[i])) + nu * (timesx[i] - timesx[i - 1]) + lamb;
             c2 = dp[i][j - 1] + math.sqrt(dist(y[j - 1], y[j])) + nu * (timesy[j] - timesy[j - 1]) + lamb;
             c3 = dp[i - 1][j - 1] + math.sqrt(dist(x[i], y[j])) + math.sqrt(dist(x[i - 1], y[j - 1])) + nu * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]));


             dp[i][j] = min(c1,c2,c3);

    return dp[xlen-1][ylen-1];


def twed_ip(x,timesx,y,timesy,lamb,nu,slope=5):

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


