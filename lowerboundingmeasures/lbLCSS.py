import numpy as np

def lbLCSS(x,y,delta,epsilon):

    xmin = np.subtract(x,epsilon);
    xmax = np.add(x,epsilon);

    sum = 0;
    for i in range(len(y)):
        wmin = max(0,i-delta);
        wmax = min(len(y)-1,i + delta);

        mind = min(xmin[wmin:wmax]);
        maxd = max(xmax[wmin:wmax]);

        if y[i] >= mind and y[i] <= maxd:
            sum = sum + 1;


    return 1 - (sum/(min(len(x),len(y))));


