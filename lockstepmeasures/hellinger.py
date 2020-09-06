def hellinger(x,y):
    if len(x) != len(y):
        return -1;
    sum = 1;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum -= (x[i] * y[i]) ** (1/2);

    if (sum < 0):
        return -1
    return 2 * (sum ** (1/2));
