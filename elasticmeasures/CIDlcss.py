import numpy as np


def CID(x,y):
    cx = np.sqrt(np.sum(np.square(np.diff(x))));
    cy = np.sqrt(np.sum(np.square(np.diff(y))));
    return (max(cx,cy)/min(cx,cy))


def lcss(x,y,epsilon,constraint=None,w=5):
    
    if (constraint == "None"):
        return lcss_n(x,y,epsilon)/CID(x,y);
    elif constraint == "Sakoe-Chiba":
        return lcss_scb(x,y,epsilon,w)/CID(x,y);
    elif constraint == "Itakura":
        return lcss_ip(x,y,epsilon,w)/CID(x,y);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return lcss_n(x,y,epsilon)/CID(x,y);


def lcss_n(x,y,epsilon):

    arr = np.zeros((len(x),len(y)));
    for i in range(len(x)):

        for j in range(0,len(y)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = arr[i][j-1] + 1;
                else:
                  cost = arr[i][j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = arr[i-1][j] + 1;
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

    arr = np.zeros((len(x),len(y)));
    for i in range(len(x)):
        wmin = max(0,i-delta);
        wmax = min(len(y)-1,i+delta);

        for j in range(int(wmin),int(wmax)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = arr[i][j-1] + 1;
                else:
                  cost = arr[i][j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = arr[i-1][j] + 1;
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
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = arr[i][j-1] + 1;
                else:
                  cost = arr[i][j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = arr[i-1][j] + 1;
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