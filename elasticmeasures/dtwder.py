import numpy as np;

def dtwder(x,y,constraint=None,w=5):
    
    x = np.diff(x);
    y = np.diff(y);



    if (constraint == "None"):
        return dtw_n(x,y);
    elif constraint == "Sakoe-Chiba":
        return dtw_scb(x,y,w);
    elif constraint == "Itakura":
        return dtw_ip(x,y,w);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return dtw_n(x,y);


def dtw_n(x,y):
    
    N = np.zeros((len(x),len(y))); 
    N[0][0] = abs(x[0] - y[0]) ** 2;

    for i in range(1,len(x)):
        N[i][0] = abs(x[i] - y[0]) ** 2 + N[i-1][0];

    for i in range(1,len(y)):
        N[0][i] = abs(x[0] - y[i]) ** 2 + N[0][i-1];


    for i in range(1,len(x)):
        for j in range(1,len(y)):
            if N[i][j] != np.Inf:
                N[i][j] = abs(x[i] - y[j]) ** 2 + min(min(N[i-1][j], N[i][j-1]), N[i-1][j-1]);

    final_dtw = N[len(x)-1][len(y)-1];


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


def dtw_scb(x,y,w=100):

    cur = np.full(len(y),np.inf);
    prev = np.full(len(y),np.inf);
    for i in range(len(x)):
        minw = max(0,i-w);
        maxw = min(len(y) ,i + w);
        temp = prev;
        prev = cur;
        cur = temp;
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