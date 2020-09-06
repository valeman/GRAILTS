import numpy as np;

def CID(x,y):
    cx = np.sqrt(np.sum(np.square(np.diff(x))));
    cy = np.sqrt(np.sum(np.square(np.diff(y))));
    return (max(cx,cy)/min(cx,cy))

def cidcidedr(x,y,m,constraint=None,w=5):
    
    if (constraint == "None"):
        return cidedr_n(x,y,m);
    elif constraint == "Sakoe-Chiba":
        return cidedr_scb(x,y,m,w);
    elif constraint == "Itakura":
        return cidedr_ip(x,y,m,w);
    else:
        warnings.warn("""No permittable constraint was entered.\n
                         Defaulting to no constraint."""
                         ,RuntimeWarning)
        return cidedr_n(x,y,m);

    return 0

def cidedr_ip(x,y,m,slope):

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
                cur[j] = -j;
            elif j == 0:
                cur[j] = -i;
            else:
                if ((abs(x[0][i] - y[0][j]) <= m) and (abs(x[1][i] - y[1][j]) <= m)):
                    s1 = 0;
                else:
                    s1 = -1;

                cur[j] = max(prev[j-1] + s1,prev[j]-1,cur[j-1] - 1);
        



    return 0 - cur[len(y[0])-1];

def cidedr_scb(x,y,m,w):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));

    for i in range(len(x)):
        minw = max(0,i-w);
        maxw = min(len(y) - 1,i+w);
        temp = prev;
        prev = cur;
        cur = temp;

        for j in range(int(minw),int(maxw)):
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = -j;
            elif j == 0:
                cur[j] = -i;
            else:
                if ((abs(x[0][i] - y[0][j]) <= m) and (abs(x[1][i] - y[1][j]) <= m)):
                    s1 = 0;
                else:
                    s1 = -1;

                cur[j] = max(prev[j-1] + s1,prev[j]-1,cur[j-1] - 1);
        



    return 0 - cur[len(y[0])-1];


def cidedr_n(x,y,m):

    df = np.zeros((len(x[0]),len(y[0])));

    for i in range(len(x[0])):
        df[i][0] = -i;

    for j in range(len(y[0])):
        df[0][i] = -j;

    for i in range(len(x[0])):
        for j in range(len(y[0])):
            if ((abs(x[0][i] - y[0][j]) <= m) and (abs(x[1][i] - y[1][j]) <= m)):
                s1 = 0;
            else:
                s1 = -1;


            df[i][j] = max(df[i-1][j-1] + s1,df[i][j-1]-1,df[i-1][j]-1);


    return 0 - df[len(x[0]) - 1][len(y[0])-1];



