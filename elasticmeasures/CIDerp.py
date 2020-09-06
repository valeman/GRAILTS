import numpy as np
import math

def CID(x,y):
    cx = np.sqrt(np.sum(np.square(np.diff(x))));
    cy = np.sqrt(np.sum(np.square(np.diff(y))));
    return (max(cx,cy)/min(cx,cy))

def erpabs(x,y,m,constraint=None,w=5):
    
    if (constraint == "None"):
        return erpabs_n(x,y,m)/CID(x,y);
    elif constraint == "Sakoe-Chiba":
        return erpabs_scb(x,y,m,w)/CID(x,y);
    elif constraint == "Itakura":
        return erpabs_ip(x,y,m,w)/CID(x,y);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return erpabs_n(x,y)/CID(x,y);

def erp(x,y,m,constraint=None,w=5):
    
    if (constraint == "None"):
        return erp_n(x,y,m);
    elif constraint == "Sakoe-Chiba":
        return erp_scb(x,y,m,w);
    elif constraint == "Itakura":
        return erp_ip(x,y,m,w);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return erp_n(x,y,m);


def erp_n(x,y,m):
    np.insert(x,0,0);
    np.insert(y,0,0);
    df = np.zeros((len(x), len(y)))


    for i in range(1,len(y)):
        df[0][i] = df[0][i-1] - pow(y[i] - m, 2);

    for i in range(1,len(x)):
        df[i][0] = df[i-1][0] - pow(x[i] - m, 2);

    df[1][1] = 0;

    for i in range(1,len(x)):
        for j in range(1,len(y)):
            df[i][j] = max(df[i - 1][j-1] - pow(x[i] - y[j],2), df[i][j-1] - pow(y[j] - m,2), df[i-1][j] - pow(x[i] - m, 2));

    return math.sqrt(0 - df[len(x) - 1][len(y) -1 ]);

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
            elif j == 0:
                cur[j] = prev[j] - pow(x[i] - m, 2);
            else:
                cur[j] = max(prev[j-1] - pow(x[i] - y[j],2), cur[j-1] - pow(y[j] - m,2), prev[j] - pow(x[i] - m,2));
        
        temp = prev;
        prev = cur;
        cur = temp;

    return math.sqrt(0 - cur[len(y)-1]);

def erp_scb(x,y,m,w):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));

    for i in range(len(x)):
        minw = max(0,i-w);
        maxw = min(len(y)-1, i + w);
        temp = prev;
        prev = cur;
        cur = prev;

        for j in range(int(minw),int(maxw)):
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = cur[j-1] - pow(y[j] - m,2);
            elif j == 0:
                cur[j] = prev[j] - pow(x[i] - m, 2);
            else:
                cur[j] = max(prev[j-1] - pow(x[i] - y[j],2), cur[j-1] - pow(y[j] - m,2), prev[j] - pow(x[i] - m,2));
        
    return math.sqrt(0 - cur[len(y)-1]);


