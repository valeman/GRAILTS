import numpy as np;

def lcss(x,y,slope,epsilon):
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

        for j in range(int(wmin),int(wmax)):
            if (i + j == 0):
                cost = 0;
            else:
                if (i == 0):
                    cost = arr[i][j-1]
                elif j == wmin:
                    cost = arr[i-1][j];
                elif (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = arr[i-1][j-1] + 1;
                elif (arr[i - 1][j] > arr[i][j - 1]):
                    cost = arr[i-1][j];
                else:
                    cost = arr[i][j-1];
            arr[i][j] = cost;

    result = arr[len(x)-1][len(y)-1];

    return 1 - result/min(len(x),len(y));