import numpy as np
import warnings
def lcss(x,y,epsilon,constraint=None,w=5):
    
    if (constraint == "None"):
        return lcss_n(x,y,epsilon);
    elif constraint == "Sakoe-Chiba":
        return lcss_scb(x,y,epsilon,w);
    elif constraint == "Itakura":
        return lcss_ip(x,y,epsilon,w);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return lcss_n(x,y,epsilon);


def lcss_n(x,y,epsilon):
    cost = 0;
    arr = np.zeros((len(x),len(y)));
    for i in range(len(x)):

        for j in range(0,len(y)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i][j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i-1][j];
            else:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = arr[i-1][j-1] + 1;
                elif (arr[i - 1][j] > arr[i][j - 1]):
                    cost = arr[i-1][j];
                else:
                    cost = arr[i][j-1];
            arr[i][j] = cost;

    result = arr[len(x)-1][len(y)-1];

    return 1 - result/min(len(x),len(y));

def lcss_scb(x,y,delta,epsilon):

    cost = 0;
    arr = np.zeros((len(x),len(y)));
    for i in range(len(x)):
        wmin = max(0,i-delta);
        wmax = min(len(y),i+delta);

        for j in range(int(wmin),int(wmax)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i][j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i-1][j];
            else:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = arr[i-1][j-1] + 1;
                elif (arr[i - 1][j] > arr[i][j - 1]):
                    cost = arr[i-1][j];
                else:
                    cost = arr[i][j-1];
            arr[i][j] = cost;

    result = arr[len(x)-1][len(y)-1];

    return 1 - result/min(len(x),len(y));


def lcss_ip(x,y,slope,epsilon):
    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));

    cost = 0;
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
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = cur[j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = prev[j];
            else:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = prev[j-1] + 1;
                elif (prev[j] > cur[j - 1]):
                    cost = prev[j];
                else:
                    cost = cur[j-1];
            cur[j] = cost;

    result = cur[len(y)-1];

    return 1 - result/min(len(x),len(y));