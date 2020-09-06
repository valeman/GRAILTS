import math

def topsoe(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    logxy = []
    for i in range(len(x)):
        logxy.append(math.log(x[i] + y[i]));
    for i in range(len(x)):
        sum += (x[i] * (math.log(2*x[i]) - logxy[i])) + (y[i] * (math.log(2*y[i]) - logxy[i]));
    return sum;
