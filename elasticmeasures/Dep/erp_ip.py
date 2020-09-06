import numpy as np;

def erp_ip(x,y,m,slope):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));


    for i in range(len(x)):
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
                cur[j] = cur[j-1] - pow(y[j] - m,2);
            elif j == minw:
                cur[j] = prev[j] - pow(x[i] - m, 2);
            else:
                cur[j] = max(prev[j-1] - pow(x[i] - y[j],2), cur[j-1] - pow(y[j] - m,2), prev[j] - pow(x[i] - m,2));
        
        temp = prev;
        prev = cur;
        cur = temp;

    return math.sqrt(0 - cur[len(y)-1]);



def erpabs_ip(x,y,m,slope):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));


    for i in range(len(x)):
        minw =  np.ceil(max(min_slope * i,
                    ((ylen-1) - max_slope * (xlen - 1)
                        + max_slope * i)))
        maxw = np.floor(min(max_slope * i,
                   ((ylen - 1) - min_slope * (xlen - 1)
                      + min_slope * i)) + 1);
        temp = prev;
        prev = cur;
        cur = temp;
        for j in range(int(minw),int(maxw)):
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = cur[j-1] - abs(y[j] - m);
            elif j == 0:
                cur[j] = prev[j] - abs(x[i] - m);
            else:
                cur[j] = max(prev[j-1] - abs(x[i] - y[j]), cur[j-1] - abs(y[j] - m), prev[j] - abs(x[i] - m));

        temp = prev;
        prev = cur;
        cur = temp;

    return math.sqrt(0 - cur[len(y)-1]);
