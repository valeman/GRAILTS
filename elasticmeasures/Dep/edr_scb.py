import numpy as np


def edr_scb(x,y,m,w):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));

    for i in range(len(x)):
        minw = max(0,i-w);
        maxw = min(len(y),i+w);
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
