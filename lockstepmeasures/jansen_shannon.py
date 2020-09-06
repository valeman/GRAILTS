import math

def jansen_shannon(x,y):
    if len(x) != len(y):
        return -1;
    logxy = [];
    for i in range(len(x)):
        if x[i] + y[i] <= 0:
            return -1;
        logxy.append(math.log(x[i] + y[i]));

    sum = 0
    for i in range(len(x)):
        if x[i] <= 0:
            return -1;
        if y[i] <= 0:
            return -1;
        sum += x[i] * (math.log(2*x[i]) - logxy[i]) + y[i] * (math.log(2*y[i]) - logxy[i])
    
    return .5 * sum;