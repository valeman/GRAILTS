def gower(x,y):
    if len(x) != len(y):
        return -1;
    if len(x) == 0:
        return -1;
    sum = 0;
    for  i in range(len(x)):
        sum += abs(x[i] - y[i]);
    return 1/len(x) * sum;
