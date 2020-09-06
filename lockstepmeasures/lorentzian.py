import math

def lorentzian(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += math.log(1 + abs(x[i] - y[i]))
    return sum;
