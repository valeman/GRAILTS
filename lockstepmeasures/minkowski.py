def minkowski(x,y,p):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += abs(x[i] - y[i]) ** p;

    return sum ** (1/p);
