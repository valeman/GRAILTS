import math

def jensen_difference(x,y):
    xyavg = [];
    for i in range(len(x)):
        if x[i] + y[i] <= 0:
            return -1;
        xyavg.append((x[i] + y[i])/2);

    sum = 0;
    for i in range(len(x)):
        if y[i] <= 0:
            return -1;
        if x[i] <= 0:
            return -1;
        sum += (x[i] * math.log(x[i]) + y[i] * math.log(y[i])) / 2 - xyavg[i] * math.log(xyavg[i]);

    return sum;

