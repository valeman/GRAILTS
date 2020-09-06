import numpy as np;

def swale_ip(x,y,p,r,epsilon,slope=5):

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
                cur[j] = 0;
            elif i == 0:
                cur[j] = j * p;
            elif j == 0:
                cur[j] = i * p;
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r;
                else:
                   cur[j] = max(prev[j], cur[j-1]) + p;
        

    return cur[len(y)-1];

