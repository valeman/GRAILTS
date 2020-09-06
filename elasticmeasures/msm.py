import numpy as np
import warnings
def msm(x,y,c,constraint=None,w=5):
    
    if (constraint == "None"):
        return msm_n(x,y,c);
    elif constraint == "Sakoe-Chiba":
        return msm_scb(x,y,c,w);
    elif constraint == "Itakura":
        return msm_ip(x,y,c,w);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return msm_n(x,y,c);

def msm_dist(new, x, y, c):
    if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
        dist = c;
    else:
        dist = c + min(abs(new - x), abs(new - y))

    return dist;

def msm_n(x,y,c):

    cost = np.zeros((len(x),len(y)));

    xlen = len(x);
    ylen = len(y);

    cost[0][0] = abs(x[0] - y[0]);

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


def msm_ip(x,y,c,slope):

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
                cur[j] = abs(x[0] - y[0]);
            elif i == 0:
                cur[j] = prev[j] + msm_dist(x[i],x[i-1],y[0],c);
            elif j == minw:
                cur[j] = cur[j-1] +  msm_dist(y[i], x[0],y[i-1],c);
            else:
                cur[j] = min(prev[j-1] + abs(x[i] - y[j]),
                            prev[j] + msm_dist(x[i], x[i -1],y[j],c),
                            cur[j-1] + msm_dist(y[j], x[i], y[j-1],c));
        
        

    return cur[ylen-1];

def msm_scb(x,y,c,w):

    xlen = len(x);
    ylen = len(y);

    prev = np.zeros(ylen);
    cur = np.zeros(ylen);


    for i in range(xlen):
        temp = prev;
        prev = cur;
        cur = temp;
        minw = max(0,i-w);
        maxw = min(len(y),i+w);

        for j in range(int(minw),int(maxw)):

            if i + j == 0:
                cur[j] = abs(x[0] - y[0]);
            elif i == 0:
                cur[j] = prev[j] + msm_dist(x[i],x[i-1],y[0],c);
            elif j == minw:
                cur[j] = cur[j-1] +  msm_dist(y[i], x[0],y[i-1],c);
            else:
                cur[j] = min(prev[j-1] + abs(x[i] - y[j]),
                            prev[j] + msm_dist(x[i], x[i -1],y[j],c),
                            cur[j-1] + msm_dist(y[j], x[i], y[j-1],c));
        
        

    return cur[ylen-1];