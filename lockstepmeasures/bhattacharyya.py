import math

def bhattacharyya(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum += math.sqrt(x[i] * y[i]);

    return - math.log(sum);