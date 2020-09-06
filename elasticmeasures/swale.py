import numpy as np;
import warnings


def swale(x,y,p,r,epsilon,constraint=None,w=5):
    
    if (constraint == "None"):
        return swale_n(x,y);
    elif constraint == "Sakoe-Chiba":
        return swale_scb(x,y,p,r,epsilon,w);
    elif constraint == "Itakura":
        return swale_ip(x,y,p,r,epsilon,w);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return swale_n(x,y);

def swale_n(x,y,p,r,epsilon):
    df = np.zeros((len(x),len(y)))

    for i in range(len(y)):
        df[0][i] = i * p;
    for i in range(len(x)):
        df[i][0] = i * p;

    for i in range(1,len(x)):
        for j in range(1,len(y)):
            if (abs(x[i] - y[i]) <= epsilon):
                df[i][j] = df[i-1][j-1] + r;
            else:
               df[i][j] = max(df[i][j-1], df[i-1][j]) + p;

    return df[len(x)-1][len(y)-1];

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
            elif j == minw:
                cur[j] = i * p;
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r;
                else:
                   cur[j] = max(prev[j], cur[j-1]) + p;
        

    return cur[len(y)-1];

def swale_scb(x,y,p,r,epsilon,w):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));


    for i in range(1,len(x)):
        temp = prev;
        prev = cur;
        cur = temp;
        minw = max(0,i-w);
        maxw = min(i+w,len(y));

        for j in range(int(minw),int(maxw)):
            
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = j * p;
            elif j == minw:
                cur[j] = i * p;
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r;
                else:
                   cur[j] = max(prev[j], cur[j-1]) + p;
        

    return cur[len(y)-1];
