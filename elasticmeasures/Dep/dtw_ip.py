import numpy as np

def dtw_ip(x,y,slope = 5):

    cur = np.full(len(y),np.inf);
    prev = np.full(len(y),np.inf);
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
                cur[j] = abs(x[0] - y[0]) ** 2;
            elif i == 0:
                cur[j] = abs(x[0] - y[j]) ** 2 + cur[j-1];
            elif j == 0:
                cur[j] = abs(x[i] - y[0]) ** 2 + prev[j];
            else:
                cur[j] = abs(x[i] - y[j]) ** 2 + min(prev[j-1],prev[j],cur[j-1]);
    final_dtw = cur[len(y)-1];

    return final_dtw ** (1/2);


