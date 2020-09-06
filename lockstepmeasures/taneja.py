import math

def taneja(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    xy = [];
    for i in range(len(x)):
        xy.append((x[i] + y[i])/ 2);
    
    for i in range(len(x)):
        sum += xy[i] * math.log(xy[i] / math.sqrt(x[i] *y[i]))

    return sum;
