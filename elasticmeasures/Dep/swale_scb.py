import numpy as np;

def swale_scb(x,y,p,r,epsilon,w):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));


    for i in range(1,len(x)):
        temp = prev;
        prev = cur;
        cur = prev;
        minw = max(0,i-w);
        maxw = min(i+w,len(y));

        for j in range(int(minw),int(maxw)):
            
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = j * p;
            elif j == 0:
                cur[j] = i * p;
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r;
                else:
                   cur[j] = max(prev[j], cur[j-1]) + p;
        

    return cur[len(y)-1];
