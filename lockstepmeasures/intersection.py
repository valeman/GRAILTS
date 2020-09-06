def intersection(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += abs(x[i] - y[i]);
    
    return 1/2 * sum;
