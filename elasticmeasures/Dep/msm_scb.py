import numpy as np

def msm_dist(new, x, y, c):
    if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
        dist = c;
    else:
        dist = c + min(abs(new - x), abs(new - y))

    return dist;

def msm(x,y,c,w):

    xlen = len(x);
    ylen = len(y);

    prev = np.zeros(ylen);
    cur = np.zeros(ylen);


    for i in range(xlen):
        temp = prev;
        prev = cur;
        cur = temp;
        minw = max(0,i-w);
        maxw = min(0,i+w);

        for j in range(int(minw),int(maxw)):

            if i + j == 0:
                cur[j] = abs(x[0] - y[0]);
            elif i == 0:
                cur[j] = prev[j] + msm_dist(x[i],x[i-1],y[0],c);
            elif j == minw:
                cur[j] = cur[j-1] +  msm_dist(y[i], x[0],y[i-1],c);
            else:
                cur[j] = min(prev[j-1] + abs(x[i] - y[j]),
                            prev[j] + msm_dist(x[i], x[i -1],y[j],c),
                            cur[j-1] + msm_dist(y[j], x[i], y[j-1],c));
        
        

    return cur[ylen-1];
