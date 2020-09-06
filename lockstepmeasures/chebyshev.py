def chebyshev(x,y):
    if len(x) != len(y):
        return -1;
    max = 0
    for i in range(len(x)):
        dif = abs(x[i] - y[i]);
        if max < dif:
            max = dif;
    return max;
